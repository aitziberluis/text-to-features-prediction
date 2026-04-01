"""
Clasificador de edad usando representaciones SAE sobre GPT-2.

Extrae las activaciones de una capa intermedia de GPT-2, las codifica
a traves de la SAE entrenada (representacion sparse), y entrena
clasificadores lineales (SGD) en multiples configuraciones.
Todo se mantiene en memoria (no se guardan activaciones ni modelos a disco).

A nivel de COMENTARIO:
  - last_token: representacion SAE del ultimo token real del comentario
  - mean: media de representaciones SAE de todos los tokens reales
  - Cada uno con y sin balanceo de clase

A nivel de USUARIO:
  - mean_of_last: media de last_token SAE de todos los comentarios del usuario
  - mean_of_mean: media de mean SAE de todos los comentarios del usuario
  - Cada uno con y sin balanceo de clase

Rangos de edad: 14_19, 20_29, 30_39, 40_plus

Evaluacion solo en eval set (test reservado para uso futuro).
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import dotenv
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

# Importar desde el directorio raiz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocesamiento import preparar_dataset_para_edad
from tiny_sae import Sae

dotenv.load_dotenv()

# =====================
# CONFIGURACION
# =====================

MODEL = "openai-community/gpt2"
CONTEXT_LEN = 512
# Forzar GPU 0 explícitamente
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    DEVICE = "cuda:0"
    print("cuda0")
else:
    DEVICE = "cpu"

# Ruta a la SAE entrenada sobre todo el corpus
PATH_SAE = "sae-ckpts/sae-gpt2-comments"

# Rutas de datos
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"
TEXT_COLUMN = "body"
MAX_COMMENTS = None

# Directorio donde se guardan las representaciones SAE extraidas (HDD)
ACTIVATIONS_DIR = "/hdd/aitziber.l/activaciones_sae_gpt2_edad"

# Rangos de edad
AGE_GROUPS = ["14_19", "20_29", "30_39", "40_plus"]
LABEL_MAP = {g: i for i, g in enumerate(AGE_GROUPS)}
NUM_CLASSES = len(AGE_GROUPS)

# Directorio compartido para los indices de split (comun a GPT y SAE)
SPLITS_DIR = "data/splits_edad"

# Splits
TEST_SIZE = 0.15
EVAL_SIZE = 0.15
RANDOM_STATE = 42

# Entrenamiento
EXTRACT_BATCH_SIZE = 32
MIN_EXTRACT_BATCH_SIZE = 4
TRAIN_EPOCHS = 1
SGD_ALPHA = 1e-5

# Precision/limpieza para reducir uso de memoria
SAE_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
FEATURES_DTYPE = np.float16
CUDA_EMPTY_CACHE_EVERY = 200

# Progreso: imprimir cada hora (3600 s)
PROGRESS_INTERVAL = 3600

# Configuraciones a correr
COMMENT_POOLINGS = ["last_token", "mean"]
USER_POOLINGS = ["mean_of_last", "mean_of_mean"]
BALANCE_CONFIGS = [
    {"name": "sin_balanceo", "use_class_weights": False},
    {"name": "balanceo_manual", "use_class_weights": True},
]

# Resampling: submuestra maxima para SMOTE/ADASYN (RAM limitada, alta dim)
MAX_RESAMPLE_TRAIN = 50_000

# Tecnicas de resampling a probar
RESAMPLE_CONFIGS = [
    {"name": "SMOTE", "cls": SMOTE, "kwargs": {"random_state": RANDOM_STATE}},
    {"name": "ADASYN", "cls": ADASYN, "kwargs": {"random_state": RANDOM_STATE}},
]


# =====================
# UTILIDADES
# =====================


def calcular_pesos_clase_manual(y: np.ndarray) -> np.ndarray:
    """Asigna pesos manuales segun ranking de frecuencia de clase.

    Clase menos frecuente -> 1.3, siguiente -> 1.1, siguiente -> 0.95,
    mas frecuente -> 0.85.
    """
    PESOS_POR_RANGO = [1.3, 1.1, 0.95, 0.85]  # de menos a mas frecuente
    counts = np.bincount(y, minlength=NUM_CLASSES)
    # Orden de clases de menos a mas frecuente
    rank_order = np.argsort(counts)  # indices ordenados por count ascendente
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    for rank, class_idx in enumerate(rank_order):
        weights[class_idx] = PESOS_POR_RANGO[rank]
    return weights


def sample_weights_from_class_weights(y: np.ndarray, class_weights: Optional[np.ndarray]) -> np.ndarray:
    """Devuelve vector de sample weights; 1.0 si class_weights es None."""
    if class_weights is None:
        return np.ones(len(y), dtype=np.float32)
    return class_weights[y]


def _is_oom_error(exc: BaseException) -> bool:
    """Detecta OOM de CUDA lanzado como excepcion tipada o RuntimeError."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


# =====================
# CARGA DE DATOS
# =====================


def cargar_datos_edad() -> pd.DataFrame:
    """Carga comentarios con edad conocida usando preprocesamiento centralizado."""
    df, _ = preparar_dataset_para_edad(
        path_comentarios=PATH_COMENTARIOS,
        path_autores=PATH_AUTORES,
        max_comments=MAX_COMMENTS,
        solo_edad_conocida=True,
    )

    if TEXT_COLUMN in df.columns and TEXT_COLUMN != "text":
        df = df.rename(columns={TEXT_COLUMN: "text"})

    df = df.dropna(subset=["text", "age_group"]).reset_index(drop=True)
    df = df[df["age_group"].isin(AGE_GROUPS)].reset_index(drop=True)

    assert set(df["age_group"].unique()) <= set(AGE_GROUPS)

    dist = df["age_group"].value_counts().to_dict()
    print(f"Rangos de edad presentes: {dist}")
    return df


# =====================
# EXTRACCION DE REPRESENTACIONES SAE
# =====================


def _pool_sparse_to_dense(
    top_acts: torch.Tensor,
    top_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    num_latents: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convierte activaciones sparse SAE (top_acts, top_indices) en vectores
    densos pooled por last_token y mean de tokens reales.

    Args:
        top_acts: (batch, seq, k)
        top_indices: (batch, seq, k)
        attention_mask: (batch, seq)
        num_latents: dimension del espacio latente SAE

    Returns:
        last_token_pooled: (batch, num_latents)
        mean_pooled: (batch, num_latents)
    """
    batch_size, seq_len, k = top_acts.shape
    device = top_acts.device
    dtype = top_acts.dtype
    mask = attention_mask.to(device).float()

    # --- last_token ---
    lengths = mask.sum(dim=1).clamp(min=1).long() - 1
    batch_idx = torch.arange(batch_size, device=device)

    last_acts = top_acts[batch_idx, lengths, :]      # (batch, k)
    last_indices = top_indices[batch_idx, lengths, :] # (batch, k)

    last_pooled = torch.zeros(batch_size, num_latents, device=device, dtype=dtype)
    last_pooled.scatter_add_(
        1,
        last_indices.long(),
        last_acts,
    )

    # --- mean (weighted by attention mask) ---
    mean_pooled = torch.zeros(batch_size, num_latents, device=device, dtype=dtype)

    # Mask per-token: (batch, seq, 1) para anular padding
    token_mask = mask.unsqueeze(-1)  # (batch, seq, 1)
    masked_acts = top_acts * token_mask  # (batch, seq, k)

    # Expandir batch dimension para scatter
    batch_ids = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, seq_len, k)

    # Flatten todo y hacer scatter_add
    flat_batch = batch_ids.reshape(-1)
    flat_indices = top_indices.reshape(-1).long()
    flat_acts = masked_acts.reshape(-1)

    mean_pooled.index_put_(
        (flat_batch, flat_indices),
        flat_acts,
        accumulate=True,
    )

    valid_tokens = mask.sum(dim=1).clamp(min=1).unsqueeze(1)  # (batch, 1)
    mean_pooled = mean_pooled / valid_tokens

    return last_pooled, mean_pooled


def _extraer_activaciones(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """Extrae representaciones SAE de GPT-2 en memoria (sin guardar a disco)."""

    n = len(df)
    print(f"\nExtrayendo representaciones SAE para {n:,} comentarios...", flush=True)
    print(f"SAE: {PATH_SAE}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"CUDA disponible: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU actual: {torch.cuda.current_device()}", flush=True)
        print(f"Nombre GPU: {torch.cuda.get_device_name()}", flush=True)

    # Cargar SAE
    sae = Sae.load_from_disk(PATH_SAE, device=DEVICE)
    if hasattr(sae, "to"):
        sae = sae.to(device=DEVICE, dtype=SAE_DTYPE)
    num_latents = sae.cfg.num_latents
    hookpoint_name = sae.cfg.hookpoint
    print(f"SAE cargada: {num_latents} latentes, k={sae.cfg.k}, hookpoint={hookpoint_name}", flush=True)
    est_features_gb = (n * num_latents * np.dtype(FEATURES_DTYPE).itemsize * 2) / (1024 ** 3)
    print(f"Features dtype: {FEATURES_DTYPE.__name__} | RAM estimada features: {est_features_gb:.1f} GiB", flush=True)

    # Cargar modelo GPT-2
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": DEVICE},
        dtype=SAE_DTYPE,
    )
    model.eval()

    hookpoint_module = model.get_submodule(hookpoint_name)
    captured = {}

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        captured["act"] = outputs.detach()

    handle = hookpoint_module.register_forward_hook(hook)

    total_steps = math.ceil(n / EXTRACT_BATCH_SIZE)

    # Pre-allocate arrays en disco con memmap
    os.makedirs(ACTIVATIONS_DIR, exist_ok=True)
    last_token_arr = np.memmap(
        os.path.join(ACTIVATIONS_DIR, "last_token.mmap"),
        dtype=FEATURES_DTYPE, mode="w+", shape=(n, num_latents),
    )
    mean_token_arr = np.memmap(
        os.path.join(ACTIVATIONS_DIR, "mean_token.mmap"),
        dtype=FEATURES_DTYPE, mode="w+", shape=(n, num_latents),
    )

    last_print = time.time()

    try:
        with torch.inference_mode():
            start = 0
            step = 0
            current_batch_size = EXTRACT_BATCH_SIZE

            while start < n:
                batch_size = current_batch_size
                while True:
                    end = min(start + batch_size, n)
                    batch_texts = df["text"].iloc[start:end].tolist()

                    try:
                        tokens = tokenizer(
                            batch_texts,
                            max_length=CONTEXT_LEN,
                            truncation=True,
                            padding="max_length",
                            return_attention_mask=True,
                            return_tensors="pt",
                        )
                        input_ids = tokens["input_ids"].to(model.device)
                        attention_mask = tokens["attention_mask"].to(model.device)

                        model.transformer(input_ids=input_ids, attention_mask=attention_mask)

                        acts = captured["act"].to(sae.dtype).to(sae.device)
                        top_acts, top_indices = sae.encode(acts)

                        last_pooled, mean_pooled = _pool_sparse_to_dense(
                            top_acts=top_acts,
                            top_indices=top_indices,
                            attention_mask=attention_mask,
                            num_latents=num_latents,
                        )

                        last_token_arr[start:end] = last_pooled.float().cpu().numpy().astype(FEATURES_DTYPE, copy=False)
                        mean_token_arr[start:end] = mean_pooled.float().cpu().numpy().astype(FEATURES_DTYPE, copy=False)
                        break
                    except Exception as e:
                        if not _is_oom_error(e):
                            raise

                        captured.clear()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        if batch_size <= MIN_EXTRACT_BATCH_SIZE:
                            raise RuntimeError(
                                f"OOM incluso con batch_size={batch_size}. "
                                f"Prueba menor CONTEXT_LEN o precision mas agresiva."
                            ) from e

                        new_batch_size = max(MIN_EXTRACT_BATCH_SIZE, batch_size // 2)
                        print(
                            f"  OOM detectado en step={step+1} ({start:,}:{end:,}). "
                            f"Reduciendo batch_size {batch_size} -> {new_batch_size} y reintentando...",
                            flush=True,
                        )
                        batch_size = new_batch_size

                # Liberar memoria explícitamente
                del batch_texts, tokens, input_ids, attention_mask, acts, top_acts, top_indices
                del last_pooled, mean_pooled
                captured.clear()
                if torch.cuda.is_available() and ((step + 1) % CUDA_EMPTY_CACHE_EVERY == 0):
                    torch.cuda.empty_cache()

                start = end
                step += 1
                current_batch_size = batch_size

                now = time.time()
                if now - last_print >= PROGRESS_INTERVAL or step == 1 or start >= n:
                    pct = 100.0 * start / n
                    print(
                        f"  [{pct:5.1f}%] step {step}/{total_steps} "
                        f"({start:,}/{n:,} comentarios) | batch={current_batch_size}",
                        flush=True,
                    )
                    last_print = now

    finally:
        handle.remove()
        # Limpiar GPU memory al finalizar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del model, tokenizer, sae
        print(f"Representaciones SAE extraidas y guardadas en {ACTIVATIONS_DIR}.", flush=True)

    labels = np.array([LABEL_MAP[g] for g in df["age_group"]], dtype=np.int8)
    authors = df["author"].to_numpy() if "author" in df.columns else None

    # Flush memmaps y guardar metadata para cache
    last_token_arr.flush()
    mean_token_arr.flush()
    meta = {"n": n, "num_latents": num_latents}
    with open(os.path.join(ACTIVATIONS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f)
    np.save(os.path.join(ACTIVATIONS_DIR, "labels.npy"), labels)
    if authors is not None:
        np.save(os.path.join(ACTIVATIONS_DIR, "authors.npy"), authors)

    return last_token_arr, mean_token_arr, labels, authors, num_latents


def extraer_activaciones(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """Extrae representaciones SAE con cache en disco (memmap).

    Returns: (last_token, mean_token, labels, authors_array, num_latents)
    """
    meta_path = os.path.join(ACTIVATIONS_DIR, "meta.json")
    if os.path.exists(meta_path):
        print(f"\nCargando representaciones SAE desde cache: {ACTIVATIONS_DIR}")
        with open(meta_path) as f:
            meta = json.load(f)
        n, num_latents = meta["n"], meta["num_latents"]

        if n != len(df):
            print(f"  AVISO: cache tiene {n} filas pero df tiene {len(df)}. Re-extrayendo...")
            return _extraer_activaciones(df)

        last_token = np.memmap(
            os.path.join(ACTIVATIONS_DIR, "last_token.mmap"),
            dtype=FEATURES_DTYPE, mode="r", shape=(n, num_latents),
        )
        mean_token = np.memmap(
            os.path.join(ACTIVATIONS_DIR, "mean_token.mmap"),
            dtype=FEATURES_DTYPE, mode="r", shape=(n, num_latents),
        )
        labels = np.load(os.path.join(ACTIVATIONS_DIR, "labels.npy"))
        authors_path = os.path.join(ACTIVATIONS_DIR, "authors.npy")
        authors = np.load(authors_path, allow_pickle=True) if os.path.exists(authors_path) else None
        print(f"  Cache cargada: {n:,} comentarios, {num_latents} latentes")
        return last_token, mean_token, labels, authors, num_latents

    return _extraer_activaciones(df)


# =====================
# SPLITS
# =====================


def dividir_comentarios(
    labels: np.ndarray, df: pd.DataFrame, authors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/eval/test a nivel comentario basado en split de usuarios.

    Todos los comentarios de un mismo usuario van al mismo split para
    evitar data leakage entre train/eval/test.
    """
    os.makedirs(SPLITS_DIR, exist_ok=True)
    split_path = os.path.join(SPLITS_DIR, "split_comentarios_por_usuario.npz")

    if os.path.exists(split_path):
        data = np.load(split_path)
        train_idx = data["train_idx"]
        eval_idx = data["eval_idx"]
        test_idx = data["test_idx"]

        all_idx = np.concatenate([train_idx, eval_idx, test_idx])
        if len(np.unique(all_idx)) == len(labels) and all_idx.min() >= 0 and all_idx.max() < len(labels):
            print(f"Cargando split de comentarios (por usuario) desde {split_path}")
            return train_idx, eval_idx, test_idx

        print("Split de comentarios en cache invalido para este dataset. Regenerando...")

    # Obtener split de usuarios
    train_auth, eval_auth, test_auth = dividir_usuarios(df)
    train_auth_set = set(train_auth)
    eval_auth_set = set(eval_auth)
    test_auth_set = set(test_auth)

    # Asignar cada comentario al split de su usuario
    train_idx = []
    eval_idx = []
    test_idx = []
    for i, auth in enumerate(authors):
        if auth in train_auth_set:
            train_idx.append(i)
        elif auth in eval_auth_set:
            eval_idx.append(i)
        elif auth in test_auth_set:
            test_idx.append(i)

    train_idx = np.array(train_idx, dtype=np.int64)
    eval_idx = np.array(eval_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)

    np.savez(split_path, train_idx=train_idx, eval_idx=eval_idx, test_idx=test_idx)
    print(f"Split de comentarios (por usuario) guardado en {split_path}")
    print(f"  Sin leakage: cada usuario aparece en un unico split.")
    return train_idx, eval_idx, test_idx


def dividir_usuarios(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/eval/test estratificado a nivel de usuario.

    Carga los autores de cada split desde SPLITS_DIR si existen.
    Si no existen, los genera y guarda.
    """
    os.makedirs(SPLITS_DIR, exist_ok=True)
    paths = {
        s: os.path.join(SPLITS_DIR, f"usuario_{s}_auth.npy")
        for s in ("train", "eval", "test")
    }

    # Intentar cargar splits existentes
    if all(os.path.exists(p) for p in paths.values()):
        train_auth = np.load(paths["train"], allow_pickle=True)
        eval_auth = np.load(paths["eval"], allow_pickle=True)
        test_auth = np.load(paths["test"], allow_pickle=True)

        user_df = df[["author", "age_group"]].drop_duplicates("author")
        total_saved = len(train_auth) + len(eval_auth) + len(test_auth)
        if total_saved == len(user_df):
            print(f"Splits de usuarios cargados desde {SPLITS_DIR}/")
            return train_auth, eval_auth, test_auth
        print(f"Num usuarios cambio ({total_saved} -> {len(user_df)}). Regenerando...")

    user_df = df[["author", "age_group"]].drop_duplicates("author")
    authors = user_df["author"].to_numpy()
    user_labels = np.array([LABEL_MAP[g] for g in user_df["age_group"]], dtype=np.int8)

    train_eval_auth, test_auth = train_test_split(
        authors, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=user_labels,
    )
    mask_te = np.isin(authors, train_eval_auth)
    user_labels_te = user_labels[mask_te]

    eval_rel = EVAL_SIZE / (1.0 - TEST_SIZE)
    train_auth, eval_auth = train_test_split(
        train_eval_auth, test_size=eval_rel, random_state=RANDOM_STATE,
        stratify=user_labels_te,
    )

    # Guardar a disco
    np.save(paths["train"], train_auth)
    np.save(paths["eval"], eval_auth)
    np.save(paths["test"], test_auth)
    print(f"Splits de usuarios guardados en {SPLITS_DIR}/")

    return train_auth, eval_auth, test_auth


# =====================
# EVALUACION
# =====================


def evaluar(nombre: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Imprime y devuelve metricas de evaluacion (multiclase)."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    all_labels = list(range(NUM_CLASSES))
    prec_c = precision_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    rec_c = recall_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    f1_c = f1_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)

    print(f"\n=== {nombre} ===")
    print(f"Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f} | F1 macro: {f1_mac:.4f} | F1 weighted: {f1_w:.4f}")
    for i, group in enumerate(AGE_GROUPS):
        print(f"  {group:>8s}: prec={prec_c[i]:.4f} rec={rec_c[i]:.4f} f1={f1_c[i]:.4f}")
    print(classification_report(y_true, y_pred, target_names=AGE_GROUPS, labels=all_labels, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=all_labels))

    result = {
        "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_mac), "f1_weighted": float(f1_w),
    }
    for i, group in enumerate(AGE_GROUPS):
        result[f"precision_{group}"] = float(prec_c[i])
        result[f"recall_{group}"] = float(rec_c[i])
        result[f"f1_{group}"] = float(f1_c[i])

    return result


# =====================
# ENTRENAMIENTO NIVEL COMENTARIO
# =====================


def entrenar_comentario(
    feats: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray,
    y_train: np.ndarray, y_eval: np.ndarray,
    class_weights: Optional[np.ndarray],
    pooling_name: str, balance_name: str,
    scaler: StandardScaler = None,
) -> Tuple[SGDClassifier, Dict]:
    """Entrena SGD incremental a nivel comentario y evalua en eval.

    Si se proporciona scaler, normaliza cada batch con el.
    """
    run_name = f"comentario_{pooling_name}_{balance_name}"
    print(f"\n{'='*60}")
    print(f"ENTRENANDO: {run_name}")
    print(f"  Train: {len(y_train):,} | Eval: {len(y_eval):,}")
    if class_weights is not None:
        cw_str = ", ".join(f"{AGE_GROUPS[i]}={class_weights[i]:.3f}" for i in range(NUM_CLASSES))
        print(f"  Pesos de clase: {cw_str}")
    else:
        print(f"  Pesos de clase: ninguno (todos 1.0)")
    print(f"{'='*60}")

    clf = SGDClassifier(
        loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
        random_state=RANDOM_STATE, average=True,
    )

    classes = np.arange(NUM_CLASSES, dtype=np.int64)
    n = len(y_train)
    batch_size = 4096
    total_steps = math.ceil(n / batch_size)
    last_print = time.time()

    for epoch in range(TRAIN_EPOCHS):
        perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(n)
        for step, start in enumerate(range(0, n, batch_size)):
            batch_perm = perm[start:start + batch_size]
            xb = np.asarray(feats[train_idx[batch_perm]], dtype=np.float32)
            if scaler is not None:
                xb = scaler.transform(xb)
            yb = y_train[batch_perm]
            sw = sample_weights_from_class_weights(yb, class_weights)

            if epoch == 0 and step == 0:
                clf.partial_fit(xb, yb, classes=classes, sample_weight=sw)
            else:
                clf.partial_fit(xb, yb, sample_weight=sw)

            now = time.time()
            if now - last_print >= PROGRESS_INTERVAL or step == total_steps - 1:
                pct = 100.0 * (step + 1) / total_steps
                print(f"  [Epoch {epoch+1}] {pct:5.1f}% ({step+1}/{total_steps})")
                last_print = now

    # Eval por lotes
    y_pred_parts = []
    for ev_start in range(0, len(y_eval), batch_size):
        ev_end = min(ev_start + batch_size, len(y_eval))
        xb = np.asarray(feats[eval_idx[ev_start:ev_end]], dtype=np.float32)
        if scaler is not None:
            xb = scaler.transform(xb)
        y_pred_parts.append(clf.predict(xb))
    y_pred = np.concatenate(y_pred_parts)
    metrics = evaluar(f"EVAL {run_name}", y_eval, y_pred)

    return clf, metrics


# =====================
# ENTRENAMIENTO NIVEL USUARIO
# =====================


def _agregar_por_usuario(
    authors: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    author_set: set,
) -> Tuple[np.ndarray, np.ndarray]:
    """Agrega features por media de usuario para un subconjunto de autores.

    Escanea features (puede ser memmap) por chunks secuenciales para evitar OOM.
    """
    num_latents = features.shape[1]

    auth_list = sorted(author_set)
    auth_to_idx = {a: i for i, a in enumerate(auth_list)}
    n_users = len(auth_list)

    user_sums = np.zeros((n_users, num_latents), dtype=np.float64)
    user_counts = np.zeros(n_users, dtype=np.int64)
    user_labels = np.full(n_users, -1, dtype=np.int64)

    row_to_user = np.full(len(authors), -1, dtype=np.int64)
    for i, auth in enumerate(authors):
        if auth in auth_to_idx:
            uidx = auth_to_idx[auth]
            row_to_user[i] = uidx
            if user_labels[uidx] == -1:
                user_labels[uidx] = labels[i]

    chunk_size = 8192
    n = len(authors)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_user_ids = row_to_user[start:end]

        valid_mask = chunk_user_ids >= 0
        if not valid_mask.any():
            continue

        chunk_feats = np.asarray(features[start:end])
        np.add.at(user_sums, chunk_user_ids[valid_mask], chunk_feats[valid_mask].astype(np.float64))
        np.add.at(user_counts, chunk_user_ids[valid_mask], 1)

    valid = user_counts > 0
    user_sums[valid] /= user_counts[valid, np.newaxis]
    return user_sums[valid].astype(np.float32), user_labels[valid]


def entrenar_usuario(
    authors: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    train_auth: np.ndarray, eval_auth: np.ndarray,
    class_weights: Optional[np.ndarray],
    pooling_name: str, balance_name: str,
) -> Tuple[SGDClassifier, Dict]:
    """Entrena SGD a nivel usuario y evalua en eval."""
    run_name = f"usuario_{pooling_name}_{balance_name}"

    print(f"\nAgregando features por usuario para {run_name}...")
    X_train, y_train = _agregar_por_usuario(authors, features, labels, set(train_auth))
    X_eval, y_eval = _agregar_por_usuario(authors, features, labels, set(eval_auth))

    print(f"\n{'='*60}")
    print(f"ENTRENANDO: {run_name}")
    print(f"  Train users: {len(y_train):,} | Eval users: {len(y_eval):,}")
    if class_weights is not None:
        cw_str = ", ".join(f"{AGE_GROUPS[i]}={class_weights[i]:.3f}" for i in range(NUM_CLASSES))
        print(f"  Pesos de clase: {cw_str}")
    else:
        print(f"  Pesos de clase: ninguno (todos 1.0)")
    print(f"{'='*60}")

    clf = SGDClassifier(
        loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
        random_state=RANDOM_STATE, average=True,
    )

    classes = np.arange(NUM_CLASSES, dtype=np.int64)
    sw = sample_weights_from_class_weights(y_train, class_weights)

    clf.partial_fit(X_train, y_train, classes=classes, sample_weight=sw)

    for epoch in range(1, TRAIN_EPOCHS):
        perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(len(y_train))
        sw_perm = sample_weights_from_class_weights(y_train[perm], class_weights)
        clf.partial_fit(X_train[perm], y_train[perm], sample_weight=sw_perm)

    y_pred = clf.predict(X_eval)
    metrics = evaluar(f"EVAL {run_name}", y_eval, y_pred)

    return clf, metrics


# =====================
# MAIN
# =====================


def main():
    print("=" * 70, flush=True)
    print("CLASIFICADOR EDAD - REPRESENTACIONES SAE SOBRE GPT-2", flush=True)
    print("=" * 70, flush=True)

    # Verificar que la SAE existe
    if not Path(PATH_SAE).exists():
        raise FileNotFoundError(
            f"No se encuentra la SAE en {PATH_SAE}. "
            "Primero ejecuta sae_gpt.py para entrenarla."
        )

    # 1. Cargar datos
    df = cargar_datos_edad()

    if "author" not in df.columns:
        print("AVISO: No hay columna 'author', se omitiran predicciones a nivel usuario.", flush=True)
        has_author = False
    else:
        df["author"] = df["author"].astype(str).str.strip()
        has_author = True

    # 2. Extraer representaciones SAE (solo en memoria, sin guardar a disco)
    last_token, mean_token, labels, authors, num_latents = extraer_activaciones(df)
    print(f"\nRepresentaciones SAE: num_latents={num_latents}, comentarios={len(labels):,}", flush=True)

    # Distribucion de labels
    for i, group in enumerate(AGE_GROUPS):
        count = int((labels == i).sum())
        print(f"  {group}: {count:,} ({100*count/len(labels):.1f}%)", flush=True)

    all_results = {}

    # ==============================
    # A) NIVEL COMENTARIO
    # ==============================
    print("\n" + "#" * 70, flush=True)
    print("# A) CLASIFICACION A NIVEL DE COMENTARIO", flush=True)
    print("#" * 70, flush=True)

    train_idx, eval_idx, test_idx = dividir_comentarios(labels, df, authors)
    y_train_c = labels[train_idx]
    y_eval_c = labels[eval_idx]

    print(f"\nSplit comentarios: train={len(train_idx):,} eval={len(eval_idx):,} test={len(test_idx):,}")
    for i, group in enumerate(AGE_GROUPS):
        tr_n = int((y_train_c == i).sum())
        ev_n = int((y_eval_c == i).sum())
        print(f"  {group}: train={tr_n:,} eval={ev_n:,}")

    # Pre-calcular pesos de clase manuales sobre el train set
    train_class_weights_manual = calcular_pesos_clase_manual(y_train_c)

    comment_features = {
        "last_token": (last_token, train_idx, eval_idx),
        "mean": (mean_token, train_idx, eval_idx),
    }

    for pooling_name in COMMENT_POOLINGS:
        feats, tr_idx, ev_idx = comment_features[pooling_name]

        # Fit scaler incremental en train (memmap grande)
        print(f"\n  Ajustando StandardScaler para {pooling_name} (incremental)...")
        scaler = StandardScaler()
        scaler_batch = 8192
        n_tr = len(tr_idx)
        for sc_start in range(0, n_tr, scaler_batch):
            sc_end = min(sc_start + scaler_batch, n_tr)
            chunk = np.asarray(feats[tr_idx[sc_start:sc_end]], dtype=np.float32)
            scaler.partial_fit(chunk)
        print(f"    Scaler ajustado sobre {n_tr:,} muestras")

        # --- Configuraciones con pesos de clase (sin resampling) ---
        for bal_cfg in BALANCE_CONFIGS:
            if not bal_cfg["use_class_weights"]:
                cw = None
            else:
                cw = train_class_weights_manual

            clf, metrics = entrenar_comentario(
                feats=feats, train_idx=tr_idx, eval_idx=ev_idx,
                y_train=y_train_c, y_eval=y_eval_c,
                class_weights=cw,
                pooling_name=pooling_name, balance_name=bal_cfg["name"],
                scaler=scaler,
            )

            run_key = f"comentario_{pooling_name}_{bal_cfg['name']}"
            all_results[run_key] = metrics

        # --- Configuraciones con resampling (SMOTE, ADASYN) ---
        n_tr = len(y_train_c)
        if n_tr > MAX_RESAMPLE_TRAIN:
            print(f"\n  Submuestreando {n_tr:,} -> {MAX_RESAMPLE_TRAIN:,} para resampling...")
            rng = np.random.RandomState(RANDOM_STATE)
            counts = np.bincount(y_train_c, minlength=NUM_CLASSES)
            fracs = counts / counts.sum()
            sub_counts = np.round(fracs * MAX_RESAMPLE_TRAIN).astype(int)
            sub_idx_parts = []
            for c in range(NUM_CLASSES):
                c_idx = np.where(y_train_c == c)[0]
                n_take = min(sub_counts[c], len(c_idx))
                sub_idx_parts.append(rng.choice(c_idx, size=n_take, replace=False))
            sub_idx = np.concatenate(sub_idx_parts)
            X_tr_sub = np.asarray(feats[tr_idx[sub_idx]], dtype=np.float32)
            y_tr_sub = y_train_c[sub_idx]
        else:
            X_tr_sub = np.asarray(feats[tr_idx], dtype=np.float32)
            y_tr_sub = y_train_c

        # Normalizar antes de resampling
        X_tr_sub_norm = scaler.transform(X_tr_sub)

        for resample_cfg in RESAMPLE_CONFIGS:
            resample_name = resample_cfg["name"]
            print(f"\n  Aplicando {resample_name} sobre {len(y_tr_sub):,} muestras...")
            for i, group in enumerate(AGE_GROUPS):
                print(f"    Antes {group}: {int((y_tr_sub==i).sum()):,}")

            try:
                sampler = resample_cfg["cls"](**resample_cfg["kwargs"])
                X_resampled, y_resampled = sampler.fit_resample(X_tr_sub_norm, y_tr_sub)
                for i, group in enumerate(AGE_GROUPS):
                    print(f"    Despues {group}: {int((y_resampled==i).sum()):,}")
            except Exception as e:
                print(f"    ERROR en {resample_name}: {e}")
                continue

            run_name = f"comentario_{pooling_name}_{resample_name}"
            print(f"\n{'='*60}")
            print(f"ENTRENANDO: {run_name}")
            print(f"  Train: {len(y_resampled):,} | Eval: {len(y_eval_c):,}")
            print(f"{'='*60}")

            clf = SGDClassifier(
                loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
                random_state=RANDOM_STATE, average=True,
            )

            classes = np.arange(NUM_CLASSES, dtype=np.int64)
            n_res = len(y_resampled)
            batch_size = 4096
            total_steps = math.ceil(n_res / batch_size)
            last_print = time.time()

            for epoch in range(TRAIN_EPOCHS):
                perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(n_res)
                for step, start_r in enumerate(range(0, n_res, batch_size)):
                    idx = perm[start_r:start_r + batch_size]
                    xb = X_resampled[idx]
                    yb = y_resampled[idx]

                    if epoch == 0 and step == 0:
                        clf.partial_fit(xb, yb, classes=classes)
                    else:
                        clf.partial_fit(xb, yb)

                    now = time.time()
                    if now - last_print >= PROGRESS_INTERVAL or step == total_steps - 1:
                        pct = 100.0 * (step + 1) / total_steps
                        print(f"  [Epoch {epoch+1}] {pct:5.1f}% ({step+1}/{total_steps})")
                        last_print = now

            y_pred_parts = []
            eval_bs = 4096
            for ev_start in range(0, len(y_eval_c), eval_bs):
                ev_end = min(ev_start + eval_bs, len(y_eval_c))
                xb = np.asarray(feats[ev_idx[ev_start:ev_end]], dtype=np.float32)
                xb = scaler.transform(xb)
                y_pred_parts.append(clf.predict(xb))
            y_pred = np.concatenate(y_pred_parts)
            metrics = evaluar(f"EVAL {run_name}", y_eval_c, y_pred)

            run_key = f"comentario_{pooling_name}_{resample_name}"
            all_results[run_key] = metrics

        # Liberar RAM de submuestra
        del X_tr_sub, y_tr_sub
        import gc; gc.collect()

    # ==============================
    # B) NIVEL USUARIO
    # ==============================
    if has_author and authors is not None:
        print("\n" + "#" * 70)
        print("# B) CLASIFICACION A NIVEL DE USUARIO")
        print("#" * 70)

        train_auth, eval_auth, test_auth = dividir_usuarios(df)
        print(f"\nSplit usuarios: train={len(train_auth):,} eval={len(eval_auth):,} test={len(test_auth):,}")

        user_features = {
            "mean_of_last": last_token,
            "mean_of_mean": mean_token,
        }

        for pooling_name in USER_POOLINGS:
            feats = user_features[pooling_name]

            # Pre-agregar features por usuario (una sola vez por pooling)
            print(f"\n  Agregando features por usuario para {pooling_name}...")
            X_u_train, y_u_train = _agregar_por_usuario(authors, feats, labels, set(train_auth))
            X_u_eval, y_u_eval = _agregar_por_usuario(authors, feats, labels, set(eval_auth))

            print(f"  Usuarios train: {len(y_u_train):,}")
            for i, group in enumerate(AGE_GROUPS):
                print(f"    {group}: {int((y_u_train==i).sum()):,}")
            print(f"  Usuarios eval:  {len(y_u_eval):,}")

            # Fit scaler en train de usuarios
            u_scaler = StandardScaler()
            u_scaler.fit(X_u_train)

            # --- Configuraciones con pesos de clase ---
            for bal_cfg in BALANCE_CONFIGS:
                if not bal_cfg["use_class_weights"]:
                    cw = None
                else:
                    cw = train_class_weights_manual

                X_tr_n = u_scaler.transform(X_u_train)
                X_ev_n = u_scaler.transform(X_u_eval)

                run_name = f"usuario_{pooling_name}_{bal_cfg['name']}"
                print(f"\n{'='*60}")
                print(f"ENTRENANDO: {run_name}")
                print(f"  Train users: {len(y_u_train):,} | Eval users: {len(y_u_eval):,}")
                if cw is not None:
                    cw_str = ", ".join(f"{AGE_GROUPS[i]}={cw[i]:.3f}" for i in range(NUM_CLASSES))
                    print(f"  Pesos de clase: {cw_str}")
                else:
                    print(f"  Pesos de clase: ninguno (todos 1.0)")
                print(f"{'='*60}")

                clf = SGDClassifier(
                    loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
                    random_state=RANDOM_STATE, average=True,
                )
                classes = np.arange(NUM_CLASSES, dtype=np.int64)
                sw = sample_weights_from_class_weights(y_u_train, cw)
                clf.partial_fit(X_tr_n, y_u_train, classes=classes, sample_weight=sw)

                for epoch in range(1, TRAIN_EPOCHS):
                    perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(len(y_u_train))
                    sw_perm = sample_weights_from_class_weights(y_u_train[perm], cw)
                    clf.partial_fit(X_tr_n[perm], y_u_train[perm], sample_weight=sw_perm)

                y_pred = clf.predict(X_ev_n)
                metrics = evaluar(f"EVAL {run_name}", y_u_eval, y_pred)
                all_results[run_name] = metrics

            # --- Configuraciones con resampling (SMOTE, ADASYN) ---
            X_tr_n = u_scaler.transform(X_u_train)
            X_ev_n = u_scaler.transform(X_u_eval)

            for resample_cfg in RESAMPLE_CONFIGS:
                resample_name = resample_cfg["name"]
                print(f"\n  Aplicando {resample_name} a nivel usuario ({len(y_u_train):,} usuarios)...")

                try:
                    sampler = resample_cfg["cls"](**resample_cfg["kwargs"])
                    X_resampled, y_resampled = sampler.fit_resample(X_tr_n, y_u_train)
                    for i, group in enumerate(AGE_GROUPS):
                        print(f"    Despues {group}: {int((y_resampled==i).sum()):,}")
                except Exception as e:
                    print(f"    ERROR en {resample_name}: {e}")
                    continue

                run_name = f"usuario_{pooling_name}_{resample_name}"
                print(f"\n{'='*60}")
                print(f"ENTRENANDO: {run_name}")
                print(f"  Train users: {len(y_resampled):,} | Eval users: {len(y_u_eval):,}")
                print(f"{'='*60}")

                clf = SGDClassifier(
                    loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
                    random_state=RANDOM_STATE, average=True,
                )
                classes = np.arange(NUM_CLASSES, dtype=np.int64)
                clf.partial_fit(X_resampled, y_resampled, classes=classes)

                for epoch in range(1, TRAIN_EPOCHS):
                    perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(len(y_resampled))
                    clf.partial_fit(X_resampled[perm], y_resampled[perm])

                y_pred = clf.predict(X_ev_n)
                metrics = evaluar(f"EVAL {run_name}", y_u_eval, y_pred)
                all_results[run_name] = metrics

    # ==============================
    # RESUMEN FINAL
    # ==============================
    print("\n\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS (EVAL)")
    print("=" * 70)
    header_f1 = " ".join(f"{'F1_'+g:>8s}" for g in AGE_GROUPS)
    print(f"{'Config':<45} {'Acc':>6} {'BalAcc':>7} {'F1mac':>6} {header_f1}")
    print("-" * (45 + 6 + 7 + 6 + 8 * NUM_CLASSES + NUM_CLASSES + 3))
    for key, m in all_results.items():
        f1_vals = " ".join(f"{m.get(f'f1_{g}', 0.0):8.4f}" for g in AGE_GROUPS)
        print(f"{key:<45} {m['accuracy']:.4f} {m['balanced_accuracy']:.5f} "
              f"{m['f1_macro']:.4f} {f1_vals}")

    # (No se guarda resumen a disco para ahorrar almacenamiento)

    print("\n" + "=" * 70)
    print("COMPLETADO - Test reservado para uso futuro")
    print("=" * 70)


if __name__ == "__main__":
    main()
