"""
Clasificador de PERCEIVING usando representaciones SAE sobre GPT-2.

Modo STREAMING: extrae las activaciones de una capa intermedia de GPT-2,
las codifica a traves de la SAE entrenada (representacion sparse), y entrena
clasificadores lineales (SGD) de forma incremental sin almacenar las
representaciones completas en disco.

A nivel de COMENTARIO:
  - last_token: representacion SAE del ultimo token real del comentario
  - mean: media de representaciones SAE de todos los tokens reales
  - Cada uno con: sin balanceo, balanceo manual, SMOTE, ADASYN

A nivel de USUARIO:
  - mean_of_last: media de last_token SAE de todos los comentarios del usuario
  - mean_of_mean: media de mean SAE de todos los comentarios del usuario
  - Cada uno con: sin balanceo, balanceo manual, SMOTE, ADASYN

Clasificacion binaria: 0 vs 1
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

# importar desde el directorio raiz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocesamiento import preparar_dataset_para_mbti
from tiny_sae import Sae

dotenv.load_dotenv()

TRAIT_NAME = "perceiving"

MODEL = "Qwen/Qwen3-4B-Base"
CONTEXT_LEN = 256  # P99 token len ~391; truncamos 2.5% (cola larga)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

# ruta a la SAE entrenada sobre todo el corpus
PATH_SAE = "sae-ckpts/qwen"

# rutas de datos
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"
TEXT_COLUMN = "body"
MAX_COMMENTS = None

# clases binarias
CLASS_NAMES = ["0", "1"]
NUM_CLASSES = 2

# directorio compartido para los indices de split (comun a GPT y SAE)
SPLITS_DIR = f"data/splits_{TRAIT_NAME}"

# splits
TEST_SIZE = 0.15
EVAL_SIZE = 0.15
RANDOM_STATE = 42

# entrenamiento
EXTRACT_BATCH_SIZE = 128
MIN_EXTRACT_BATCH_SIZE = 4
TRAIN_EPOCHS = 1
SGD_ALPHA = 1e-5

# precision/limpieza para reducir uso de memoria
SAE_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
CUDA_EMPTY_CACHE_EVERY = 200

# progreso: imprimir cada hora (3600 s)
PROGRESS_INTERVAL = 300

# configuraciones a correr
COMMENT_POOLINGS = ["last_token", "mean"]
USER_POOLINGS = ["mean_of_last", "mean_of_mean"]
BALANCE_CONFIGS = [
    {"name": "sin_balanceo", "use_class_weights": False},
    {"name": "balanceo_manual", "use_class_weights": True},
    {"name": "undersampling", "use_class_weights": False},
]

def calcular_pesos_clase_manual(y: np.ndarray) -> np.ndarray:
    """Asigna pesos manuales: clase menos frecuente -> 1.1, mas frecuente -> 0.9."""
    PESOS_POR_RANGO = [1.1, 0.9]  # de menos a mas frecuente
    counts = np.bincount(y, minlength=NUM_CLASSES)
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

def random_undersample(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE):
    """Submuestrea aleatoriamente cada clase al tamano de la clase minoritaria."""
    rng = np.random.RandomState(random_state)
    classes = np.arange(NUM_CLASSES)
    counts = np.bincount(y, minlength=NUM_CLASSES)
    min_count = counts[counts > 0].min()
    print(f"Undersampling: min_count={min_count:,} (de {dict(zip(classes, counts))})")
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

def random_undersample_mask(y: np.ndarray, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Devuelve mascara booleana con los indices submuestreados."""
    rng = np.random.RandomState(random_state)
    classes = np.arange(NUM_CLASSES)
    counts = np.bincount(y, minlength=NUM_CLASSES)
    min_count = counts[counts > 0].min()
    print(f"Undersampling: min_count={min_count:,} (de {dict(zip(classes, counts))})")
    mask = np.zeros(len(y), dtype=bool)
    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) == 0:
            continue
        chosen = rng.choice(c_idx, size=min_count, replace=False)
        mask[chosen] = True
    return mask

def cargar_datos() -> pd.DataFrame:
    """Carga comentarios con TRAIT_NAME conocido usando preprocesamiento centralizado."""
    df, _ = preparar_dataset_para_mbti(
        path_comentarios=PATH_COMENTARIOS,
        path_autores=PATH_AUTORES,
        columna_mbti=TRAIT_NAME,
        max_comments=MAX_COMMENTS,
    )

    if TEXT_COLUMN in df.columns and TEXT_COLUMN != "text":
        df = df.rename(columns={TEXT_COLUMN: "text"})

    df = df.dropna(subset=["text", TRAIT_NAME]).reset_index(drop=True)
    df["label"] = df[TRAIT_NAME].astype(int)

    dist = df["label"].value_counts().to_dict()
    print(f"Distribucion de {TRAIT_NAME}: {dist}")
    return df

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
    dtype = torch.float32
    top_acts = top_acts.to(torch.float32)
    mask = attention_mask.to(device=device, dtype=dtype)

    # last_token
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

    # mean (weighted by attention mask)
    mean_pooled = torch.zeros(batch_size, num_latents, device=device, dtype=dtype)

    # mask per-token: (batch, seq, 1) para anular padding
    token_mask = mask.unsqueeze(-1)  # (batch, seq, 1)
    masked_acts = top_acts * token_mask  # (batch, seq, k)

    # expandir batch dimension para scatter
    batch_ids = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, seq_len, k)

    # flatten todo y hacer scatter_add
    flat_batch = batch_ids.reshape(-1)
    flat_indices = top_indices.reshape(-1).long()
    flat_acts = masked_acts.reshape(-1).to(mean_pooled.dtype)

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
    print(f"SAE cargada: {num_latents} latentes, k={sae.cfg.k}, hookpoint={hookpoint_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # critico para que `last real token = lengths-1`: GPT-2 ya por defecto es
    # right-padded, pero lo fijamos explicitamente.
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, device_map={"": DEVICE}, dtype=SAE_DTYPE,
    )
    model.eval()

    # OPT #1: truncar el modelo despues del HOOKPOINT. Las capas posteriores
    # son compute desperdiciado (solo consumimos la activacion del hook).
    from _clasificador_utils_qwen import truncar_modelo_para_hookpoint
    _trunk, _, _keep = truncar_modelo_para_hookpoint(model, hookpoint_name)
    model._trunk = _trunk
    print(f"Modelo truncado a las primeras {_keep} capas para hook '{hookpoint_name}'")

    # OPT #3: torch.compile sobre el bloque transformer. dynamic=True por
    # las shapes variables; try/except para no romper el run si falla.
    if torch.cuda.is_available():
        try:
            model._trunk = torch.compile(model._trunk, dynamic=True)
            print("torch.compile activado (dynamic=True)")
        except Exception as _ce:
            print(f"torch.compile no disponible, sigo sin compilar: {_ce}")

    hookpoint_module = model.get_submodule(hookpoint_name)
    return tokenizer, model, sae, hookpoint_module, num_latents

def _stream_sae_features(df, tokenizer, model, sae, hookpoint_module, num_latents, pass_name=""):
    """Generador que extrae representaciones SAE en streaming.

    Yields: (start, end, last_pooled_f32, mean_pooled_f32) por cada batch.
    No almacena nada en disco.
    """
    n = len(df)
    captured = {}

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        captured["act"] = outputs.detach()

    handle = hookpoint_module.register_forward_hook(hook)
    last_print = time.time()
    step = 0
    total_steps = math.ceil(n / EXTRACT_BATCH_SIZE)
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
                            texts, max_length=CONTEXT_LEN, truncation=True,
                            padding=True, return_attention_mask=True,
                            return_tensors="pt",
                        )
                        input_ids = tokens["input_ids"].to(model.device)
                        attn_mask = tokens["attention_mask"].to(model.device)

                        model._trunk(input_ids=input_ids, attention_mask=attn_mask)

                        acts = captured["act"].to(sae.dtype).to(sae.device)
                        top_acts, top_indices = sae.encode(acts)

                        last_pooled, mean_pooled = _pool_sparse_to_dense(
                            top_acts, top_indices, attn_mask, num_latents,
                        )

                        last_np = np.nan_to_num(last_pooled.float().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
                        mean_np = np.nan_to_num(mean_pooled.float().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)

                        del tokens, input_ids, attn_mask, acts
                        del top_acts, top_indices, last_pooled, mean_pooled
                        captured.clear()
                        break
                    except Exception as e:
                        if not _is_oom_error(e):
                            raise
                        captured.clear()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if bs <= MIN_EXTRACT_BATCH_SIZE:
                            raise
                        new_bs = max(MIN_EXTRACT_BATCH_SIZE, bs // 2)
                        print(f"OOM: batch {bs} -> {new_bs}", flush=True)
                        bs = new_bs

                if torch.cuda.is_available() and (step + 1) % CUDA_EMPTY_CACHE_EVERY == 0:
                    torch.cuda.empty_cache()

                step += 1
                current_bs = bs

                now = time.time()
                if now - last_print >= PROGRESS_INTERVAL or step == 1 or end >= n:
                    pct = 100.0 * end / n
                    print(
                        f"  [{pass_name} {pct:5.1f}%] step {step}/{total_steps} "
                        f"({end:,}/{n:,}) | batch={current_bs}",
                        flush=True,
                    )
                    last_print = now

                yield start, end, last_np, mean_np
                start = end
    finally:
        handle.remove()
        captured.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def dividir_usuarios(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/eval/test estratificado a nivel de usuario."""
    os.makedirs(SPLITS_DIR, exist_ok=True)
    paths = {
        s: os.path.join(SPLITS_DIR, f"usuario_{s}_auth.npy")
        for s in ("train", "eval", "test")
    }

    # intentar cargar splits existentes
    if all(os.path.exists(p) for p in paths.values()):
        train_auth = np.load(paths["train"], allow_pickle=True)
        eval_auth = np.load(paths["eval"], allow_pickle=True)
        test_auth = np.load(paths["test"], allow_pickle=True)

        user_df = df[["author", "label"]].drop_duplicates("author")
        total_saved = len(train_auth) + len(eval_auth) + len(test_auth)
        if total_saved == len(user_df):
            print(f"Splits de usuarios cargados desde {SPLITS_DIR}/")
            return train_auth, eval_auth, test_auth
        print(f"Num usuarios cambio ({total_saved} -> {len(user_df)}). Regenerando")

    user_df = df[["author", "label"]].drop_duplicates("author")
    authors = user_df["author"].to_numpy()
    user_labels = user_df["label"].to_numpy().astype(np.int8)

    train_eval_auth, test_auth = train_test_split(
        authors, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=user_labels,
    )
    # recalcular labels para sub-split
    mask_te = np.isin(authors, train_eval_auth)
    user_labels_te = user_labels[mask_te]

    eval_rel = EVAL_SIZE / (1.0 - TEST_SIZE)
    train_auth, eval_auth = train_test_split(
        train_eval_auth, test_size=eval_rel, random_state=RANDOM_STATE,
        stratify=user_labels_te,
    )

    # guardar a disco
    np.save(paths["train"], train_auth)
    np.save(paths["eval"], eval_auth)
    np.save(paths["test"], test_auth)
    print(f"Splits de usuarios guardados en {SPLITS_DIR}/")

    return train_auth, eval_auth, test_auth

def evaluar(nombre: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Imprime y devuelve metricas de evaluacion (binario)."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    all_labels = list(range(NUM_CLASSES))
    prec_c = precision_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    rec_c = recall_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    f1_c = f1_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)

    print(f"\n{nombre}")
    print(
        f"Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f} | "
        f"Precision macro: {prec_macro:.4f} | Recall macro: {rec_macro:.4f} | "
        f"F1 macro: {f1_mac:.4f} | F1 weighted: {f1_w:.4f}"
    )
    for i, cn in enumerate(CLASS_NAMES):
        print(f"{cn:>8s}: prec={prec_c[i]:.4f} rec={rec_c[i]:.4f} f1={f1_c[i]:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, labels=all_labels, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=all_labels))

    result = {
        "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
        "precision_macro": float(prec_macro), "recall_macro": float(rec_macro),
        "f1_macro": float(f1_mac), "f1_weighted": float(f1_w),
    }
    for i, cn in enumerate(CLASS_NAMES):
        result[f"precision_{cn}"] = float(prec_c[i])
        result[f"recall_{cn}"] = float(rec_c[i])
        result[f"f1_{cn}"] = float(f1_c[i])

    return result

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

# MAIN
def main():
    print(f"CLASIFICADOR {TRAIT_NAME.upper()} - SAE SOBRE GPT-2 (STREAMING)")

    if not Path(PATH_SAE).exists():
        raise FileNotFoundError(
            f"No se encuentra la SAE en {PATH_SAE}. "
            "Primero ejecuta sae_gpt.py para entrenarla."
        )

    # 1. Cargar datos
    df = cargar_datos()

    has_author = "author" in df.columns
    if has_author:
        df["author"] = df["author"].astype(str).str.strip()

    # 2. Split de usuarios
    train_auth, eval_auth, test_auth = dividir_usuarios(df)
    train_auth_set = set(train_auth)
    eval_auth_set = set(eval_auth)
    test_auth_set = set(test_auth)

    # 3. Separar train y eval DataFrames (por usuario, sin leakage)
    is_train = df["author"].isin(train_auth_set)
    is_eval = df["author"].isin(eval_auth_set)

    df_train = df[is_train].reset_index(drop=True)
    df_eval = df[is_eval].reset_index(drop=True)
    df_test = df[df["author"].isin(test_auth_set)].reset_index(drop=True)
    y_train = df_train["label"].values.astype(np.int8)
    y_eval = df_eval["label"].values.astype(np.int8)
    y_test_comments = df_test["label"].values.astype(np.int8)
    authors_train = df_train["author"].values if has_author else None
    authors_eval = df_eval["author"].values if has_author else None
    authors_test = df_test["author"].values if has_author else None

    del df
    gc.collect()

    print(f"\nComentarios: train={len(df_train):,} eval={len(df_eval):,} test={len(df_test):,}")
    for i, cn in enumerate(CLASS_NAMES):
        print(f"{cn}: train={int((y_train==i).sum()):,} eval={int((y_eval==i).sum()):,} test={int((y_test_comments==i).sum()):,}")

    train_class_weights_manual = calcular_pesos_clase_manual(y_train)

    # 4. Cargar modelos GPU
    tokenizer, model, sae, hookpoint_module, num_latents = _setup_models()
    print(f"num_latents={num_latents}")

    # 5. Preparar acumuladores
    classes = np.arange(NUM_CLASSES, dtype=np.int64)

    # scalers (se ajustan online durante pass A)
    scalers = {"last_token": StandardScaler(), "mean": StandardScaler()}

    # SGD classifiers (sin_balanceo + balanceo_manual + undersampling, entrenados online)
    clf_comment = {}
    for pooling in COMMENT_POOLINGS:
        for bal_cfg in BALANCE_CONFIGS:
            key = (pooling, bal_cfg["name"])
            clf_comment[key] = SGDClassifier(
                loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
                random_state=RANDOM_STATE, average=True,
            )

    # acumuladores de usuario (online: {author: [sum_vec, count, label]})
    user_sums_train = {"last_token": {}, "mean": {}}
    user_sums_eval = {"last_token": {}, "mean": {}}

    # pre-computar mascara de undersampling para entrenamiento
    us_mask = random_undersample_mask(y_train)
    us_first_batch = True
    print(f"Undersampling: {int(us_mask.sum()):,} muestras seleccionadas de {len(y_train):,}")

    # PASS A: Stream train -> scaler + SGD + user agg + subsample
    print("PASS A: Streaming datos de entrenamiento")

    first_batch = True
    for start, end, last_np, mean_np in _stream_sae_features(
        df_train, tokenizer, model, sae, hookpoint_module, num_latents,
        pass_name="TRAIN",
    ):
        bs = end - start
        yb = y_train[start:end]
        feats_by_pooling = {"last_token": last_np, "mean": mean_np}

        # ajustar scalers incrementalmente
        scalers["last_token"].partial_fit(last_np)
        scalers["mean"].partial_fit(mean_np)

        # mascara de undersampling para este batch
        batch_us_mask = us_mask[start:end]

        # escalar y entrenar SGD online
        for pooling in COMMENT_POOLINGS:
            feats_scaled = scalers[pooling].transform(
                feats_by_pooling[pooling].astype(np.float32)
            )
            for bal_cfg in BALANCE_CONFIGS:
                key = (pooling, bal_cfg["name"])
                if bal_cfg["name"] == "undersampling":
                    # solo entrenar con filas undersampled
                    if batch_us_mask.any():
                        xb_us = feats_scaled[batch_us_mask]
                        yb_us = yb[batch_us_mask]
                        if us_first_batch:
                            clf_comment[key].partial_fit(
                                xb_us, yb_us, classes=classes,
                            )
                        else:
                            clf_comment[key].partial_fit(xb_us, yb_us)
                else:
                    cw = train_class_weights_manual if bal_cfg["use_class_weights"] else None
                    sw = sample_weights_from_class_weights(yb, cw)
                    if first_batch:
                        clf_comment[key].partial_fit(
                            feats_scaled, yb, classes=classes, sample_weight=sw,
                        )
                    else:
                        clf_comment[key].partial_fit(feats_scaled, yb, sample_weight=sw)

        if batch_us_mask.any():
            us_first_batch = False

        # acumular features de usuario (train)
        if has_author:
            batch_auth = authors_train[start:end]
            for pooling in COMMENT_POOLINGS:
                d = user_sums_train[pooling]
                feats = feats_by_pooling[pooling]
                for i in range(bs):
                    auth = batch_auth[i]
                    if auth not in d:
                        d[auth] = [
                            np.zeros(num_latents, dtype=np.float64),
                            0,
                            int(yb[i]),
                        ]
                    entry = d[auth]
                    entry[0] += feats[i].astype(np.float64)
                    entry[1] += 1

        first_batch = False
        del last_np, mean_np

    print(f"\nPass A completado: {len(y_train):,} muestras procesadas.")

    # todos los clasificadores de nivel comentario
    all_clf = dict(clf_comment)
    trained_runs = {}

    # PASS B: Stream eval -> predicciones + user agg
    print("PASS B: Streaming datos de evaluacion")

    eval_preds = {key: [] for key in all_clf}

    for start, end, last_np, mean_np in _stream_sae_features(
        df_eval, tokenizer, model, sae, hookpoint_module, num_latents,
        pass_name="EVAL",
    ):
        bs = end - start
        feats_by_pooling = {"last_token": last_np, "mean": mean_np}

        for key, clf in all_clf.items():
            pooling = key[0]
            feats_scaled = scalers[pooling].transform(
                feats_by_pooling[pooling].astype(np.float32)
            )
            eval_preds[key].append(clf.predict(feats_scaled))

        # acumular features de usuario (eval)
        if has_author:
            yb = y_eval[start:end]
            batch_auth = authors_eval[start:end]
            for pooling in COMMENT_POOLINGS:
                d = user_sums_eval[pooling]
                feats = feats_by_pooling[pooling]
                for i in range(bs):
                    auth = batch_auth[i]
                    if auth not in d:
                        d[auth] = [
                            np.zeros(num_latents, dtype=np.float64),
                            0,
                            int(yb[i]),
                        ]
                    entry = d[auth]
                    entry[0] += feats[i].astype(np.float64)
                    entry[1] += 1

        del last_np, mean_np

    all_results = {}
    print("A) CLASIFICACION A NIVEL DE COMENTARIO")

    for key in all_clf:
        pooling, balance = key
        run_name = f"comentario_{pooling}_{balance}"
        y_pred = np.concatenate(eval_preds[key])
        metrics = evaluar(f"EVAL {run_name}", y_eval, y_pred)
        all_results[run_name] = metrics
        trained_runs[run_name] = {
            "level": "comentario",
            "pooling": pooling,
            "clf": all_clf[key],
            "scaler": scalers[pooling],
        }

    del eval_preds
    gc.collect()

    if has_author:
        print("B) CLASIFICACION A NIVEL DE USUARIO")

        for user_pooling in USER_POOLINGS:
            comment_pooling = "last_token" if user_pooling == "mean_of_last" else "mean"

            X_u_train, y_u_train = _build_user_arrays(user_sums_train[comment_pooling], num_latents)
            X_u_eval, y_u_eval = _build_user_arrays(user_sums_eval[comment_pooling], num_latents)

            print(f"\n  Usuarios {user_pooling}: train={len(y_u_train):,} eval={len(y_u_eval):,}")
            for i, cn in enumerate(CLASS_NAMES):
                print(
                    f"    {cn}: train={int((y_u_train==i).sum()):,} "
                    f"eval={int((y_u_eval==i).sum()):,}"
                )

            u_scaler = StandardScaler()
            u_scaler.fit(X_u_train)
            X_tr_n = u_scaler.transform(X_u_train)
            X_ev_n = u_scaler.transform(X_u_eval)

            # configuraciones de balanceo
            for bal_cfg in BALANCE_CONFIGS:
                if bal_cfg["name"] == "undersampling":
                    X_u_us, y_u_us = random_undersample(X_u_train, y_u_train)
                    X_tr_n_cur = u_scaler.transform(X_u_us)
                    cw = None
                else:
                    cw = train_class_weights_manual if bal_cfg["use_class_weights"] else None
                    X_tr_n_cur = X_tr_n
                    X_u_us, y_u_us = X_u_train, y_u_train

                run_name = f"usuario_{user_pooling}_{bal_cfg['name']}"
                print(f"ENTRENANDO: {run_name}")
                print(f"Train users: {len(y_u_us):,} | Eval users: {len(y_u_eval):,}")

                clf = SGDClassifier(
                    loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
                    random_state=RANDOM_STATE, average=True,
                )
                sw = sample_weights_from_class_weights(y_u_us, cw)
                clf.partial_fit(X_tr_n_cur, y_u_us, classes=classes, sample_weight=sw)

                for epoch in range(1, TRAIN_EPOCHS):
                    perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(
                        len(y_u_us)
                    )
                    sw_perm = sample_weights_from_class_weights(y_u_us[perm], cw)
                    clf.partial_fit(X_tr_n_cur[perm], y_u_us[perm], sample_weight=sw_perm)

                y_pred = clf.predict(X_ev_n)
                metrics = evaluar(f"EVAL {run_name}", y_u_eval, y_pred)
                all_results[run_name] = metrics
                trained_runs[run_name] = {
                    "level": "usuario",
                    "pooling": user_pooling,
                    "comment_pooling": comment_pooling,
                    "clf": clf,
                    "scaler": u_scaler,
                }

            del X_u_train, y_u_train, X_u_eval, y_u_eval, X_tr_n, X_ev_n
            gc.collect()

    best_run, best_eval_metrics = _select_best_run(all_results)
    best_artifact = trained_runs[best_run]

    print("MEJOR MODELO EN EVAL")
    print(
        f"{best_run} | F1 macro={best_eval_metrics['f1_macro']:.4f} | "
        f"Recall macro={best_eval_metrics['recall_macro']:.4f} | "
        f"Precision macro={best_eval_metrics['precision_macro']:.4f}"
    )

    from _clasificador_utils import (
        select_best_per_level,
        print_best_per_level_eval,
        print_best_per_level_test,
    )
    best_per_level = select_best_per_level(all_results)
    print_best_per_level_eval(best_per_level)

    print("PASS C: Streaming datos de test (mejor por nivel)")

    best_comment_entry = best_per_level.get("comentario")
    best_user_entry = best_per_level.get("usuario")
    comment_artifact = trained_runs[best_comment_entry[0]] if best_comment_entry is not None else None
    user_artifact = trained_runs[best_user_entry[0]] if best_user_entry is not None else None
    user_comment_pooling = user_artifact.get("comment_pooling") if user_artifact is not None else None

    comment_test_preds: List[np.ndarray] = []
    user_sums_test: Dict[str, List[object]] = {}
    for start, end, last_np, mean_np in _stream_sae_features(
        df_test, tokenizer, model, sae, hookpoint_module, num_latents, pass_name="TEST"
    ):
        feats_by_pooling = {"last_token": last_np, "mean": mean_np}

        if comment_artifact is not None:
            pooling = comment_artifact["pooling"]
            X_test = comment_artifact["scaler"].transform(feats_by_pooling[pooling].astype(np.float32))
            comment_test_preds.append(comment_artifact["clf"].predict(X_test))

        if user_artifact is not None:
            batch_auth = authors_test[start:end]
            batch_labels = y_test_comments[start:end]
            feats = feats_by_pooling[user_comment_pooling]
            for i in range(end - start):
                auth = batch_auth[i]
                if auth not in user_sums_test:
                    user_sums_test[auth] = [np.zeros(num_latents, dtype=np.float64), 0, int(batch_labels[i])]
                entry = user_sums_test[auth]
                entry[0] += feats[i].astype(np.float64)
                entry[1] += 1

    best_per_level_test: Dict[str, Optional[Tuple[str, Dict[str, float]]]] = {
        "comentario": None,
        "usuario": None,
    }

    if comment_artifact is not None:
        y_test_pred_c = np.concatenate(comment_test_preds)
        m_c = evaluar(f"TEST {best_comment_entry[0]}", y_test_comments, y_test_pred_c)
        best_per_level_test["comentario"] = (best_comment_entry[0], m_c)
        all_results[f"test_{best_comment_entry[0]}"] = m_c

    if user_artifact is not None:
        X_u_test, y_u_test = _build_user_arrays(user_sums_test, num_latents)
        X_u_test = user_artifact["scaler"].transform(X_u_test)
        y_u_pred = user_artifact["clf"].predict(X_u_test)
        m_u = evaluar(f"TEST {best_user_entry[0]}", y_u_test, y_u_pred)
        best_per_level_test["usuario"] = (best_user_entry[0], m_u)
        all_results[f"test_{best_user_entry[0]}"] = m_u

    print_best_per_level_test(best_per_level_test)

    best_test_metrics = (
        best_per_level_test.get(best_artifact["level"])[1]
        if best_per_level_test.get(best_artifact["level"]) is not None
        else None
    )

    del model, tokenizer, sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print("RESUMEN DE RESULTADOS (EVAL)")
    header_f1 = " ".join(f"{'F1_'+cn:>8s}" for cn in CLASS_NAMES)
    print(f"{'Config':<45} {'Acc':>6} {'BalAcc':>7} {'F1mac':>6} {header_f1}")
    print("-" * (45 + 6 + 7 + 6 + 8 * NUM_CLASSES + NUM_CLASSES + 3))
    for key, m in all_results.items():
        f1_vals = " ".join(f"{m.get(f'f1_{cn}', 0.0):8.4f}" for cn in CLASS_NAMES)
        print(
            f"{key:<45} {m['accuracy']:.4f} {m['balanced_accuracy']:.5f} "
            f"{m['f1_macro']:.4f} {f1_vals}"
        )

    summary_path = os.path.join(OUTPUT_DIR, "resultados_resumen.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
            "best_per_level": {
                level: (
                    None
                    if best_per_level.get(level) is None
                    else {
                        "name": best_per_level[level][0],
                        "eval_metrics": best_per_level[level][1],
                        "test_metrics": (
                            best_per_level_test[level][1]
                            if best_per_level_test.get(level) is not None
                            else None
                        ),
                    }
                )
                for level in ("comentario", "usuario")
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResumen guardado en: {summary_path}")

    print("COMPLETADO - Mejor modelo evaluado tambien en test")

if __name__ == "__main__":
    main()
