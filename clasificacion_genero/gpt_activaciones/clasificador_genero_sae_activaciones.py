"""
Clasificador de genero usando representaciones SAE sobre GPT-2.

Extrae las activaciones de una capa intermedia de GPT-2, las codifica
a traves de la SAE entrenada (representacion sparse), las guarda a disco
para no repetir el forward pass, y entrena clasificadores lineales (SGD)
en multiples configuraciones:

A nivel de COMENTARIO:
  - last_token: representacion SAE del ultimo token real del comentario
  - mean: media de representaciones SAE de todos los tokens reales
  - Cada uno con y sin balanceo de clase

A nivel de USUARIO:
  - mean_of_last: media de last_token SAE de todos los comentarios del usuario
  - mean_of_mean: media de mean SAE de todos los comentarios del usuario
  - Cada uno con y sin balanceo de clase

Evaluacion solo en eval set (test reservado para uso futuro).
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dotenv
import numpy as np
import pandas as pd
import torch
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
from preprocesamiento import preparar_dataset_para_sae
from tiny_sae import Sae

dotenv.load_dotenv()

# =====================
# CONFIGURACION
# =====================

MODEL = "openai-community/gpt2"
CONTEXT_LEN = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Ruta a la SAE entrenada sobre todo el corpus
PATH_SAE = "sae-ckpts/sae-gpt2-comments"

# Rutas de datos
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"
TEXT_COLUMN = "body"
MAX_COMMENTS = None

# Directorio donde se guardan las representaciones SAE extraidas (HDD)
ACTIVATIONS_DIR = "/hdd/aitziber.l/activaciones_sae_gpt2_genero"

# Splits
TEST_SIZE = 0.15
EVAL_SIZE = 0.15
RANDOM_STATE = 42
SPLITS_DIR = "data/splits_genero_70_15_15"

# Entrenamiento
EXTRACT_BATCH_SIZE = 32
MIN_EXTRACT_BATCH_SIZE = 4
TRAIN_EPOCHS = 1
SGD_ALPHA = 1e-5

# Precision/limpieza para reducir uso de memoria
SAE_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
CUDA_EMPTY_CACHE_EVERY = 200

# Balanceo suave
FEMALE_WEIGHT_BALANCED = 1.1
MALE_WEIGHT_BALANCED = 0.9
FEMALE_WEIGHT_DEFAULT = 1.0
MALE_WEIGHT_DEFAULT = 1.0

# Progreso: imprimir cada hora (3600 s)
PROGRESS_INTERVAL = 3600

# Configuraciones a correr
COMMENT_POOLINGS = ["last_token", "mean"]
USER_POOLINGS = ["mean_of_last", "mean_of_mean"]
BALANCE_CONFIGS = [
    {"name": "sin_balanceo", "female_w": FEMALE_WEIGHT_DEFAULT, "male_w": MALE_WEIGHT_DEFAULT},
    {"name": "balanceado", "female_w": FEMALE_WEIGHT_BALANCED, "male_w": MALE_WEIGHT_BALANCED},
    {"name": "undersampling", "female_w": 1.0, "male_w": 1.0},
]

NUM_CLASSES = 2


def random_undersample(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    """Submuestrea aleatoriamente cada clase al tamaño de la clase minoritaria."""
    rng = np.random.RandomState(random_state)
    classes = np.arange(NUM_CLASSES)
    counts = np.bincount(y, minlength=NUM_CLASSES)
    min_count = counts[counts > 0].min()
    print(f"    Undersampling: min_count={min_count:,} (de {dict(zip(classes, counts))})")
    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) == 0:
            continue
        chosen = rng.choice(c_idx, size=min_count, replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def random_undersample_idx(y: np.ndarray, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Devuelve indices submuestreados al tamaño de la clase minoritaria."""
    rng = np.random.RandomState(random_state)
    classes = np.arange(NUM_CLASSES)
    counts = np.bincount(y, minlength=NUM_CLASSES)
    min_count = counts[counts > 0].min()
    print(f"    Undersampling: min_count={min_count:,} (de {dict(zip(classes, counts))})")
    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) == 0:
            continue
        chosen = rng.choice(c_idx, size=min_count, replace=False)
        indices.append(chosen)
    indices = np.concatenate(indices)
    rng.shuffle(indices)
    return indices


def random_undersample_mask(y: np.ndarray, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Devuelve mascara booleana con los indices submuestreados."""
    rng = np.random.RandomState(random_state)
    classes = np.arange(NUM_CLASSES)
    counts = np.bincount(y, minlength=NUM_CLASSES)
    min_count = counts[counts > 0].min()
    print(f"    Undersampling: min_count={min_count:,} (de {dict(zip(classes, counts))})")
    mask = np.zeros(len(y), dtype=bool)
    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) == 0:
            continue
        chosen = rng.choice(c_idx, size=min_count, replace=False)
        mask[chosen] = True
    return mask


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


# =====================
# CARGA DE DATOS
# =====================


def cargar_datos_genero() -> pd.DataFrame:
    """Carga comentarios con genero m/f usando preprocesamiento centralizado."""
    df, _ = preparar_dataset_para_sae(
        path_comentarios=PATH_COMENTARIOS,
        path_autores=PATH_AUTORES,
        max_comments=MAX_COMMENTS,
        solo_genero_conocido=True,
    )

    if TEXT_COLUMN in df.columns and TEXT_COLUMN != "text":
        df = df.rename(columns={TEXT_COLUMN: "text"})

    df = df.dropna(subset=["text", "gender_clean"]).reset_index(drop=True)
    df = df[df["gender_clean"].isin(["m", "f"])].reset_index(drop=True)

    assert set(df["gender_clean"].unique()) <= {"m", "f"}

    dist = df["gender_clean"].value_counts().to_dict()
    print(f"Generos presentes: {dist}")
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
    # Expandir sparse a dense por token, despues promediar
    # Para eficiencia, acumulamos directamente con scatter_add
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


def _setup_models():
    """Carga tokenizer, GPT-2 y SAE. Devuelve los componentes para streaming."""
    sae = Sae.load_from_disk(PATH_SAE, device=DEVICE)
    if hasattr(sae, "to"):
        sae = sae.to(device=DEVICE, dtype=SAE_DTYPE)
    num_latents = sae.cfg.num_latents
    hookpoint_name = sae.cfg.hookpoint
    print(f"SAE cargada: {num_latents} latentes, k={sae.cfg.k}, hookpoint={hookpoint_name}")

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
    return tokenizer, model, sae, hookpoint_module, num_latents


def _stream_sae_features(df, tokenizer, model, sae, hookpoint_module, num_latents, pass_name=""):
    """Generador que extrae representaciones SAE en streaming."""
    n = len(df)
    captured = {}

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        captured["act"] = outputs.detach()

    handle = hookpoint_module.register_forward_hook(hook)
    last_print = time.time()
    step = 0
    total_steps = max(1, math.ceil(n / EXTRACT_BATCH_SIZE))
    current_bs = EXTRACT_BATCH_SIZE

    try:
        with torch.inference_mode():
            start = 0
            while start < n:
                bs = current_bs
                while True:
                    end = min(start + bs, n)
                    texts = df["text"].iloc[start:end].tolist()
                    try:
                        tokens = tokenizer(
                            texts,
                            max_length=CONTEXT_LEN,
                            truncation=True,
                            padding="max_length",
                            return_attention_mask=True,
                            return_tensors="pt",
                        )
                        input_ids = tokens["input_ids"].to(model.device)
                        attn_mask = tokens["attention_mask"].to(model.device)

                        model.transformer(input_ids=input_ids, attention_mask=attn_mask)

                        acts = captured["act"].to(sae.dtype).to(sae.device)
                        top_acts, top_indices = sae.encode(acts)

                        last_pooled, mean_pooled = _pool_sparse_to_dense(
                            top_acts, top_indices, attn_mask, num_latents,
                        )

                        last_np = last_pooled.float().cpu().numpy()
                        mean_np = mean_pooled.float().cpu().numpy()

                        del tokens, input_ids, attn_mask, acts
                        del top_acts, top_indices, last_pooled, mean_pooled
                        captured.clear()
                        break
                    except Exception as exc:
                        if not _is_oom_error(exc):
                            raise
                        captured.clear()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if bs <= MIN_EXTRACT_BATCH_SIZE:
                            raise RuntimeError(
                                f"OOM incluso con batch_size={bs}. Prueba menor CONTEXT_LEN o precision mas agresiva."
                            ) from exc
                        new_bs = max(MIN_EXTRACT_BATCH_SIZE, bs // 2)
                        print(f"  OOM: batch {bs} -> {new_bs}")
                        bs = new_bs

                if torch.cuda.is_available() and (step + 1) % CUDA_EMPTY_CACHE_EVERY == 0:
                    torch.cuda.empty_cache()

                step += 1
                current_bs = bs

                now = time.time()
                if now - last_print >= PROGRESS_INTERVAL or step == 1 or end >= n:
                    pct = 100.0 * end / max(1, n)
                    print(
                        f"  [{pass_name} {pct:5.1f}%] step {step}/{total_steps} "
                        f"({end:,}/{n:,}) | batch={current_bs}"
                    )
                    last_print = now

                yield start, end, last_np, mean_np
                start = end
    finally:
        handle.remove()
        captured.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _extraer_activaciones(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """Extrae representaciones SAE de GPT-2 en memoria (sin guardar a disco)."""

    n = len(df)
    print(f"\nExtrayendo representaciones SAE para {n:,} comentarios...")
    print(f"SAE: {PATH_SAE}")

    # Cargar SAE
    sae = Sae.load_from_disk(PATH_SAE, device=DEVICE)
    num_latents = sae.cfg.num_latents
    hookpoint_name = sae.cfg.hookpoint
    print(f"SAE cargada: {num_latents} latentes, k={sae.cfg.k}, hookpoint={hookpoint_name}")

    # Cargar modelo GPT-2
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": DEVICE},
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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

    # Pre-allocate arrays en disco con memmap en float16 (~225 GB por array)
    os.makedirs(ACTIVATIONS_DIR, exist_ok=True)
    last_token_arr = np.memmap(
        os.path.join(ACTIVATIONS_DIR, "last_token.mmap"),
        dtype=np.float16, mode="w+", shape=(n, num_latents),
    )
    mean_token_arr = np.memmap(
        os.path.join(ACTIVATIONS_DIR, "mean_token.mmap"),
        dtype=np.float16, mode="w+", shape=(n, num_latents),
    )

    textos = df["text"].tolist()
    last_print = time.time()

    try:
        with torch.no_grad():
            for step, start in enumerate(range(0, n, EXTRACT_BATCH_SIZE)):
                end = min(start + EXTRACT_BATCH_SIZE, n)
                batch_texts = textos[start:end]

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

                model(input_ids=input_ids, attention_mask=attention_mask)

                acts = captured["act"].to(sae.dtype).to(sae.device)
                top_acts, top_indices = sae.encode(acts)

                last_pooled, mean_pooled = _pool_sparse_to_dense(
                    top_acts=top_acts,
                    top_indices=top_indices,
                    attention_mask=attention_mask,
                    num_latents=num_latents,
                )

                last_token_arr[start:end] = last_pooled.half().cpu().numpy()
                mean_token_arr[start:end] = mean_pooled.half().cpu().numpy()

                now = time.time()
                if now - last_print >= PROGRESS_INTERVAL or step == 0 or step == total_steps - 1:
                    pct = 100.0 * (step + 1) / total_steps
                    print(f"  [{pct:5.1f}%] step {step+1}/{total_steps} "
                          f"({end:,}/{n:,} comentarios)")
                    last_print = now

    finally:
        handle.remove()

    # Flush memmaps a disco
    last_token_arr.flush()
    mean_token_arr.flush()

    labels = np.array([0 if g == "f" else 1 for g in df["gender_clean"]], dtype=np.int8)
    authors = df["author"].to_numpy() if "author" in df.columns else None

    # Guardar metadata, labels y authors para cache
    meta = {"n": n, "num_latents": num_latents}
    with open(os.path.join(ACTIVATIONS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f)
    np.save(os.path.join(ACTIVATIONS_DIR, "labels.npy"), labels)
    if authors is not None:
        np.save(os.path.join(ACTIVATIONS_DIR, "authors.npy"), authors)

    print(f"Representaciones SAE extraidas y guardadas en {ACTIVATIONS_DIR}.")
    return last_token_arr, mean_token_arr, labels, authors, num_latents


def extraer_activaciones(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """Extrae representaciones SAE con cache en disco (memmap).

    Si los archivos memmap ya existen, los carga directamente sin
    repetir la extraccion.

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
            dtype=np.float16, mode="r", shape=(n, num_latents),
        )
        mean_token = np.memmap(
            os.path.join(ACTIVATIONS_DIR, "mean_token.mmap"),
            dtype=np.float16, mode="r", shape=(n, num_latents),
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
    """Train/eval/test estratificado a nivel usuario con cache compartido."""
    os.makedirs(SPLITS_DIR, exist_ok=True)
    split_path = os.path.join(SPLITS_DIR, "split_usuarios.npz")

    user_df = df[["author", "gender_clean"]].drop_duplicates("author")
    authors = user_df["author"].to_numpy()
    user_labels = np.array([0 if g == "f" else 1 for g in user_df["gender_clean"]], dtype=np.int8)

    if os.path.exists(split_path):
        data = np.load(split_path, allow_pickle=True)
        train_auth = data["train_auth"]
        eval_auth = data["eval_auth"]
        test_auth = data["test_auth"]

        cache_auth = np.concatenate([train_auth, eval_auth, test_auth])
        if len(np.unique(cache_auth)) == len(authors) and np.isin(cache_auth, authors).all():
            print(f"Cargando split de usuarios compartido desde {split_path}")
            return train_auth, eval_auth, test_auth

        print("Split de usuarios en cache invalido para este dataset. Regenerando...")

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

    np.savez(split_path, train_auth=train_auth, eval_auth=eval_auth, test_auth=test_auth)
    print(f"Split de usuarios guardado en {split_path}")
    return train_auth, eval_auth, test_auth


# =====================
# EVALUACION
# =====================


def evaluar(nombre: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Imprime y devuelve metricas de evaluacion."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    prec_c = precision_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    rec_c = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    f1_c = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)

    print(f"\n=== {nombre} ===")
    print(
        f"Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f} | "
        f"Precision macro: {prec_macro:.4f} | Recall macro: {rec_macro:.4f} | "
        f"F1 macro: {f1_mac:.4f} | F1 weighted: {f1_w:.4f}"
    )
    print(f"  female: prec={prec_c[0]:.4f} rec={rec_c[0]:.4f} f1={f1_c[0]:.4f}")
    print(f"  male:   prec={prec_c[1]:.4f} rec={rec_c[1]:.4f} f1={f1_c[1]:.4f}")
    print(classification_report(y_true, y_pred, target_names=["female", "male"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
        "precision_macro": float(prec_macro), "recall_macro": float(rec_macro),
        "f1_macro": float(f1_mac), "f1_weighted": float(f1_w),
        "precision_female": float(prec_c[0]), "recall_female": float(rec_c[0]), "f1_female": float(f1_c[0]),
        "precision_male": float(prec_c[1]), "recall_male": float(rec_c[1]), "f1_male": float(f1_c[1]),
    }


def _selection_score(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        float(metrics.get("f1_macro", float("-inf"))),
        float(metrics.get("recall_macro", float("-inf"))),
        float(metrics.get("precision_macro", float("-inf"))),
    )


def _select_best_run(all_results: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    best_name, best_metrics = max(
        all_results.items(),
        key=lambda item: (_selection_score(item[1]), item[0]),
    )
    return best_name, best_metrics


def _build_user_arrays(user_dict: Dict[str, List[object]], num_latents: int) -> Tuple[np.ndarray, np.ndarray]:
    users = sorted(user_dict.keys())
    n_users = len(users)
    X = np.zeros((n_users, num_latents), dtype=np.float32)
    y = np.zeros(n_users, dtype=np.int64)
    for i, auth in enumerate(users):
        s, c, lab = user_dict[auth]
        X[i] = (s / max(c, 1)).astype(np.float32)
        y[i] = lab
    return X, y


# =====================
# ENTRENAMIENTO NIVEL COMENTARIO
# =====================


def entrenar_comentario(
    feats: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray,
    y_train: np.ndarray, y_eval: np.ndarray,
    female_w: float, male_w: float,
    pooling_name: str, balance_name: str,
    scaler: StandardScaler = None,
) -> Tuple[SGDClassifier, Dict]:
    """Entrena SGD incremental a nivel comentario y evalua en eval.

    feats puede ser un np.memmap; se accede por lotes para evitar OOM.
    Si se proporciona scaler, normaliza cada batch con el.
    """
    run_name = f"comentario_{pooling_name}_{balance_name}"
    print(f"\n{'='*60}")
    print(f"ENTRENANDO: {run_name}")
    print(f"  Train: {len(y_train):,} | Eval: {len(y_eval):,}")
    print(f"  Pesos: female={female_w}, male={male_w}")
    print(f"{'='*60}")

    clf = SGDClassifier(
        loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
        random_state=RANDOM_STATE, average=True,
    )

    classes = np.array([0, 1], dtype=np.int64)
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
            sw = np.where(yb == 0, female_w, male_w).astype(np.float32)

            if epoch == 0 and step == 0:
                clf.partial_fit(xb, yb, classes=classes, sample_weight=sw)
            else:
                clf.partial_fit(xb, yb, sample_weight=sw)

            now = time.time()
            if now - last_print >= PROGRESS_INTERVAL or step == total_steps - 1:
                pct = 100.0 * (step + 1) / total_steps
                print(f"  [Epoch {epoch+1}] {pct:5.1f}% ({step+1}/{total_steps})")
                last_print = now

    # Eval (en lotes para evitar OOM con memmaps grandes)
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

    # Mapear cada fila a su indice de usuario (-1 si no esta en author_set)
    row_to_user = np.full(len(authors), -1, dtype=np.int64)
    for i, auth in enumerate(authors):
        if auth in auth_to_idx:
            uidx = auth_to_idx[auth]
            row_to_user[i] = uidx
            if user_labels[uidx] == -1:
                user_labels[uidx] = labels[i]

    # Escaneo secuencial por chunks
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
    female_w: float, male_w: float,
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
    print(f"  Pesos: female={female_w}, male={male_w}")
    print(f"{'='*60}")

    clf = SGDClassifier(
        loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
        random_state=RANDOM_STATE, average=True,
    )

    classes = np.array([0, 1], dtype=np.int64)
    sw = np.where(y_train == 0, female_w, male_w).astype(np.float32)

    clf.partial_fit(X_train, y_train, classes=classes, sample_weight=sw)

    for epoch in range(1, TRAIN_EPOCHS):
        perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(len(y_train))
        sw_perm = np.where(y_train[perm] == 0, female_w, male_w).astype(np.float32)
        clf.partial_fit(X_train[perm], y_train[perm], sample_weight=sw_perm)

    y_pred = clf.predict(X_eval)
    metrics = evaluar(f"EVAL {run_name}", y_eval, y_pred)

    return clf, metrics


# =====================
# MAIN
# =====================


def main():
    print("=" * 70)
    print("CLASIFICADOR GENERO - REPRESENTACIONES SAE SOBRE GPT-2")
    print("=" * 70)

    # Verificar que la SAE existe
    if not Path(PATH_SAE).exists():
        raise FileNotFoundError(
            f"No se encuentra la SAE en {PATH_SAE}. "
            "Primero ejecuta sae_gpt.py para entrenarla."
        )

    # 1. Cargar datos
    df = cargar_datos_genero()

    if "author" not in df.columns:
        print("AVISO: No hay columna 'author', se omitiran predicciones a nivel usuario.")
        has_author = False
    else:
        df["author"] = df["author"].astype(str).str.strip()
        has_author = True

    train_auth, eval_auth, test_auth = dividir_usuarios(df)
    train_auth_set = set(train_auth)
    eval_auth_set = set(eval_auth)
    test_auth_set = set(test_auth)

    df_train = df[df["author"].isin(train_auth_set)].reset_index(drop=True)
    df_eval = df[df["author"].isin(eval_auth_set)].reset_index(drop=True)
    df_test = df[df["author"].isin(test_auth_set)].reset_index(drop=True)
    y_train = np.array([0 if g == "f" else 1 for g in df_train["gender_clean"]], dtype=np.int8)
    y_eval = np.array([0 if g == "f" else 1 for g in df_eval["gender_clean"]], dtype=np.int8)
    y_test_comments = np.array([0 if g == "f" else 1 for g in df_test["gender_clean"]], dtype=np.int8)
    authors_train = df_train["author"].values if has_author else None
    authors_eval = df_eval["author"].values if has_author else None
    authors_test = df_test["author"].values if has_author else None

    del df
    gc.collect()

    print(f"\nComentarios: train={len(df_train):,} eval={len(df_eval):,} test={len(df_test):,}")
    print(f"  female: train={int((y_train==0).sum()):,} eval={int((y_eval==0).sum()):,} test={int((y_test_comments==0).sum()):,}")
    print(f"  male:   train={int((y_train==1).sum()):,} eval={int((y_eval==1).sum()):,} test={int((y_test_comments==1).sum()):,}")

    tokenizer, model, sae, hookpoint_module, num_latents = _setup_models()
    print(f"\nRepresentaciones SAE en streaming: num_latents={num_latents}")

    all_results = {}
    trained_runs = {}
    classes = np.array([0, 1], dtype=np.int64)
    scalers = {"last_token": StandardScaler(), "mean": StandardScaler()}
    clf_comment = {}
    for pooling in COMMENT_POOLINGS:
        for bal_cfg in BALANCE_CONFIGS:
            key = (pooling, bal_cfg["name"])
            clf_comment[key] = SGDClassifier(
                loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
                random_state=RANDOM_STATE, average=True,
            )

    user_sums_train = {"last_token": {}, "mean": {}}
    user_sums_eval = {"last_token": {}, "mean": {}}
    us_mask = random_undersample_mask(y_train)
    us_first_batch = True

    # ==============================
    # A) NIVEL COMENTARIO
    # ==============================
    print("\n" + "#" * 70)
    print("# A) CLASIFICACION A NIVEL DE COMENTARIO")
    print("#" * 70)

    first_batch = True
    for start, end, last_np, mean_np in _stream_sae_features(
        df_train, tokenizer, model, sae, hookpoint_module, num_latents, pass_name="TRAIN"
    ):
        bs = end - start
        yb = y_train[start:end]
        feats_by_pooling = {"last_token": last_np, "mean": mean_np}

        scalers["last_token"].partial_fit(last_np)
        scalers["mean"].partial_fit(mean_np)
        batch_us_mask = us_mask[start:end]

        for pooling in COMMENT_POOLINGS:
            feats_scaled = scalers[pooling].transform(feats_by_pooling[pooling].astype(np.float32))
            for bal_cfg in BALANCE_CONFIGS:
                key = (pooling, bal_cfg["name"])
                if bal_cfg["name"] == "undersampling":
                    if batch_us_mask.any():
                        xb_us = feats_scaled[batch_us_mask]
                        yb_us = yb[batch_us_mask]
                        if us_first_batch:
                            clf_comment[key].partial_fit(xb_us, yb_us, classes=classes)
                        else:
                            clf_comment[key].partial_fit(xb_us, yb_us)
                else:
                    sw = np.where(yb == 0, bal_cfg["female_w"], bal_cfg["male_w"]).astype(np.float32)
                    if first_batch:
                        clf_comment[key].partial_fit(feats_scaled, yb, classes=classes, sample_weight=sw)
                    else:
                        clf_comment[key].partial_fit(feats_scaled, yb, sample_weight=sw)

            if has_author:
                batch_auth = authors_train[start:end]
                feats = feats_by_pooling[pooling]
                user_dict = user_sums_train[pooling]
                for i in range(bs):
                    auth = batch_auth[i]
                    if auth not in user_dict:
                        user_dict[auth] = [np.zeros(num_latents, dtype=np.float64), 0, int(yb[i])]
                    entry = user_dict[auth]
                    entry[0] += feats[i].astype(np.float64)
                    entry[1] += 1

        if batch_us_mask.any():
            us_first_batch = False
        first_batch = False
        del last_np, mean_np

    print(f"\nPass A completado: {len(y_train):,} muestras procesadas.")

    eval_preds = {key: [] for key in clf_comment}
    print("\n" + "#" * 70)
    print("# PASS B: Streaming datos de evaluacion")
    print("#" * 70)

    for start, end, last_np, mean_np in _stream_sae_features(
        df_eval, tokenizer, model, sae, hookpoint_module, num_latents, pass_name="EVAL"
    ):
        bs = end - start
        feats_by_pooling = {"last_token": last_np, "mean": mean_np}

        for key, clf in clf_comment.items():
            pooling = key[0]
            feats_scaled = scalers[pooling].transform(feats_by_pooling[pooling].astype(np.float32))
            eval_preds[key].append(clf.predict(feats_scaled))

        if has_author:
            yb = y_eval[start:end]
            batch_auth = authors_eval[start:end]
            for pooling in COMMENT_POOLINGS:
                feats = feats_by_pooling[pooling]
                user_dict = user_sums_eval[pooling]
                for i in range(bs):
                    auth = batch_auth[i]
                    if auth not in user_dict:
                        user_dict[auth] = [np.zeros(num_latents, dtype=np.float64), 0, int(yb[i])]
                    entry = user_dict[auth]
                    entry[0] += feats[i].astype(np.float64)
                    entry[1] += 1

        del last_np, mean_np

    for key in clf_comment:
        pooling, balance = key
        run_name = f"comentario_{pooling}_{balance}"
        y_pred = np.concatenate(eval_preds[key])
        metrics = evaluar(f"EVAL {run_name}", y_eval, y_pred)
        all_results[run_name] = metrics
        trained_runs[run_name] = {
            "level": "comentario",
            "pooling": pooling,
            "clf": clf_comment[key],
            "scaler": scalers[pooling],
        }

    del eval_preds
    gc.collect()

    # ==============================
    # B) NIVEL USUARIO
    # ==============================
    if has_author:
        print("\n" + "#" * 70)
        print("# B) CLASIFICACION A NIVEL DE USUARIO")
        print("#" * 70)

        print(f"\nSplit usuarios: train={len(train_auth):,} eval={len(eval_auth):,} test={len(test_auth):,}")

        for pooling_name in USER_POOLINGS:
            comment_pooling = "last_token" if pooling_name == "mean_of_last" else "mean"
            print(f"\n  Agregando features por usuario para {pooling_name}...")
            X_u_train, y_u_train = _build_user_arrays(user_sums_train[comment_pooling], num_latents)
            X_u_eval, y_u_eval = _build_user_arrays(user_sums_eval[comment_pooling], num_latents)

            print(f"  Usuarios train: {len(y_u_train):,} (f={int((y_u_train==0).sum()):,} m={int((y_u_train==1).sum()):,})")
            print(f"  Usuarios eval:  {len(y_u_eval):,}")

            # Fit scaler en train de usuarios
            u_scaler = StandardScaler()
            u_scaler.fit(X_u_train)

            # --- Configuraciones de balanceo ---
            for bal_cfg in BALANCE_CONFIGS:
                if bal_cfg["name"] == "undersampling":
                    X_u_us, y_u_us = random_undersample(X_u_train, y_u_train)
                    X_tr_n = u_scaler.transform(X_u_us)
                    X_ev_n = u_scaler.transform(X_u_eval)
                else:
                    X_tr_n = u_scaler.transform(X_u_train)
                    X_ev_n = u_scaler.transform(X_u_eval)
                    X_u_us, y_u_us = X_u_train, y_u_train

                run_name = f"usuario_{pooling_name}_{bal_cfg['name']}"
                print(f"\n{'='*60}")
                print(f"ENTRENANDO: {run_name}")
                print(f"  Train users: {len(y_u_us):,} | Eval users: {len(y_u_eval):,}")
                print(f"  Pesos: female={bal_cfg['female_w']}, male={bal_cfg['male_w']}")
                print(f"{'='*60}")

                clf = SGDClassifier(
                    loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
                    random_state=RANDOM_STATE, average=True,
                )
                classes = np.array([0, 1], dtype=np.int64)
                sw = np.where(y_u_us == 0, bal_cfg["female_w"], bal_cfg["male_w"]).astype(np.float32)
                clf.partial_fit(X_tr_n, y_u_us, classes=classes, sample_weight=sw)

                for epoch in range(1, TRAIN_EPOCHS):
                    perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(len(y_u_us))
                    sw_perm = np.where(y_u_us[perm] == 0, bal_cfg["female_w"], bal_cfg["male_w"]).astype(np.float32)
                    clf.partial_fit(X_tr_n[perm], y_u_us[perm], sample_weight=sw_perm)

                y_pred = clf.predict(X_ev_n)
                metrics = evaluar(f"EVAL {run_name}", y_u_eval, y_pred)
                all_results[run_name] = metrics
                trained_runs[run_name] = {
                    "level": "usuario",
                    "pooling": pooling_name,
                    "comment_pooling": comment_pooling,
                    "clf": clf,
                    "scaler": u_scaler,
                }

    best_run, best_eval_metrics = _select_best_run(all_results)
    best_artifact = trained_runs[best_run]

    print("\n" + "=" * 70)
    print("MEJOR MODELO EN EVAL")
    print("=" * 70)
    print(
        f"{best_run} | F1 macro={best_eval_metrics['f1_macro']:.4f} | "
        f"Recall macro={best_eval_metrics['recall_macro']:.4f} | "
        f"Precision macro={best_eval_metrics['precision_macro']:.4f}"
    )

    print("\n" + "#" * 70)
    print("# PASS C: Streaming datos de test del mejor modelo")
    print("#" * 70)

    if best_artifact["level"] == "comentario":
        test_preds = []
        for _start, _end, last_np, mean_np in _stream_sae_features(
            df_test, tokenizer, model, sae, hookpoint_module, num_latents, pass_name="TEST"
        ):
            test_feats = last_np if best_artifact["pooling"] == "last_token" else mean_np
            X_test = best_artifact["scaler"].transform(test_feats.astype(np.float32))
            test_preds.append(best_artifact["clf"].predict(X_test))
        y_test = y_test_comments
        y_test_pred = np.concatenate(test_preds)
    else:
        user_sums_test = {}
        for start, end, last_np, mean_np in _stream_sae_features(
            df_test, tokenizer, model, sae, hookpoint_module, num_latents, pass_name="TEST"
        ):
            batch_auth = authors_test[start:end]
            batch_labels = y_test_comments[start:end]
            feats = last_np if best_artifact["comment_pooling"] == "last_token" else mean_np
            for i in range(end - start):
                auth = batch_auth[i]
                if auth not in user_sums_test:
                    user_sums_test[auth] = [np.zeros(num_latents, dtype=np.float64), 0, int(batch_labels[i])]
                entry = user_sums_test[auth]
                entry[0] += feats[i].astype(np.float64)
                entry[1] += 1
        X_u_test, y_test = _build_user_arrays(user_sums_test, num_latents)
        X_u_test = best_artifact["scaler"].transform(X_u_test)
        y_test_pred = best_artifact["clf"].predict(X_u_test)

    best_test_metrics = evaluar(f"TEST {best_run}", y_test, y_test_pred)

    del model, tokenizer, sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ==============================
    # RESUMEN FINAL
    # ==============================
    print("\n\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS (EVAL)")
    print("=" * 70)
    print(f"{'Config':<50} {'Acc':>6} {'BalAcc':>7} {'F1mac':>6} {'F1_f':>6} {'F1_m':>6}")
    print("-" * 85)
    for key, m in all_results.items():
        print(f"{key:<50} {m['accuracy']:.4f} {m['balanced_accuracy']:.5f} "
              f"{m['f1_macro']:.4f} {m['f1_female']:.4f} {m['f1_male']:.4f}")

    summary_path = os.path.join(OUTPUT_DIR, "resultados_resumen.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "selection_metric_order": ["f1_macro", "recall_macro", "precision_macro"],
            "eval_results": all_results,
            "best_run_on_eval": {
                "name": best_run,
                "level": best_artifact["level"],
                "pooling": best_artifact["pooling"],
                "eval_metrics": best_eval_metrics,
                "test_metrics": best_test_metrics,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResumen guardado en: {summary_path}")

    print("\n" + "=" * 70)
    print("COMPLETADO - Mejor modelo evaluado tambien en test")
    print("=" * 70)


if __name__ == "__main__":
    main()
