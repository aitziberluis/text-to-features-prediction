"""
Clasificador de THINKING usando activaciones directas de GPT-2 (NO SAE).

Extrae las activaciones de una capa intermedia de GPT-2, las guarda a disco
para no repetir el forward pass, y entrena clasificadores lineales (SGD)
en multiples configuraciones:

A nivel de COMENTARIO:
  - last_token: activacion del ultimo token real del comentario
  - mean: media de activaciones de todos los tokens reales
  - Cada uno con: sin balanceo, balanceo manual, SMOTE, ADASYN

A nivel de USUARIO:
  - mean_of_last: media de last_token de todos los comentarios del usuario
  - mean_of_mean: media de mean de todos los comentarios del usuario
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
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

# Importar desde el directorio raiz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocesamiento import preparar_dataset_para_mbti

dotenv.load_dotenv()

# =====================
# CONFIGURACION
# =====================

TRAIT_NAME = "thinking"

MODEL = "openai-community/gpt2"
CONTEXT_LEN = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HOOKPOINT = "transformer.h.8"

# Rutas
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"
TEXT_COLUMN = "body"
MAX_COMMENTS = None

# Clases binarias
CLASS_NAMES = ["0", "1"]
NUM_CLASSES = 2

# Directorio donde se guardan las activaciones extraidas
ACTIVATIONS_DIR = f"data/activaciones_gpt2_{TRAIT_NAME}"

# Directorio compartido para los indices de split (comun a GPT y SAE)
SPLITS_DIR = f"data/splits_{TRAIT_NAME}"

# Splits
TEST_SIZE = 0.15
EVAL_SIZE = 0.15
RANDOM_STATE = 42

# Entrenamiento
EXTRACT_BATCH_SIZE = 32
TRAIN_EPOCHS = 1
SGD_ALPHA = 1e-5

# Progreso: imprimir cada hora (3600 s)
PROGRESS_INTERVAL = 3600

# Configuraciones a correr
COMMENT_POOLINGS = ["last_token", "mean"]
USER_POOLINGS = ["mean_of_last", "mean_of_mean"]
BALANCE_CONFIGS = [
    {"name": "sin_balanceo", "use_class_weights": False},
    {"name": "balanceo_manual", "use_class_weights": True},
    {"name": "undersampling", "use_class_weights": False},
]

SCALER_BATCH_SIZE = 8192
USER_AGG_CHUNK_SIZE = 8192

# Output
OUTPUT_DIR = f"modelos/{TRAIT_NAME}_gpt_activaciones"


# =====================
# UTILIDADES
# =====================


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


def random_undersample(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE):
    """Submuestrea aleatoriamente cada clase al tamano de la clase minoritaria."""
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
    """Devuelve indices submuestreados al tamano de la clase minoritaria."""
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


# =====================
# CARGA DE DATOS
# =====================


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


# =====================
# EXTRACCION DE ACTIVACIONES
# =====================


def _extraer_y_guardar_activaciones(df: pd.DataFrame) -> None:
    """Extrae activaciones de GPT-2 y guarda last_token + mean por comentario."""

    n = len(df)
    print(f"\nExtrayendo activaciones de GPT-2 para {n:,} comentarios...")
    print(f"Hookpoint: {HOOKPOINT}, batch_size: {EXTRACT_BATCH_SIZE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": DEVICE},
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    hookpoint_module = model.get_submodule(HOOKPOINT)
    captured = {}

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        captured["act"] = outputs.detach()

    handle = hookpoint_module.register_forward_hook(hook)

    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd")
    total_steps = math.ceil(n / EXTRACT_BATCH_SIZE)

    # Pre-allocate arrays en disco con memmap para no ocupar toda la RAM
    os.makedirs(ACTIVATIONS_DIR, exist_ok=True)
    last_token_path = os.path.join(ACTIVATIONS_DIR, "last_token.npy")
    mean_token_path = os.path.join(ACTIVATIONS_DIR, "mean_token.npy")

    last_token_mmap = np.lib.format.open_memmap(
        last_token_path, mode="w+", dtype=np.float32, shape=(n, hidden_size)
    )
    mean_token_mmap = np.lib.format.open_memmap(
        mean_token_path, mode="w+", dtype=np.float32, shape=(n, hidden_size)
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

                acts = captured["act"].float().cpu()  # (bs, seq, hidden)
                mask = attention_mask.cpu().float()    # (bs, seq)

                # Last token: posicion del ultimo token real
                lengths = mask.sum(dim=1).clamp(min=1).long() - 1
                batch_idx = torch.arange(acts.shape[0])
                last_tok = acts[batch_idx, lengths, :]  # (bs, hidden)

                # Mean token: media sobre tokens reales
                mask_3d = mask.unsqueeze(-1)  # (bs, seq, 1)
                mean_tok = (acts * mask_3d).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

                last_token_mmap[start:end] = last_tok.numpy()
                mean_token_mmap[start:end] = mean_tok.numpy()

                now = time.time()
                if now - last_print >= PROGRESS_INTERVAL or step == 0 or step == total_steps - 1:
                    pct = 100.0 * (step + 1) / total_steps
                    print(f"  [{pct:5.1f}%] step {step+1}/{total_steps} "
                          f"({end:,}/{n:,} comentarios)")
                    last_print = now

    finally:
        handle.remove()

    # Flush memmap
    del last_token_mmap, mean_token_mmap

    # Guardar labels y authors
    labels = df["label"].to_numpy().astype(np.int8)
    np.save(os.path.join(ACTIVATIONS_DIR, "labels.npy"), labels)

    if "author" in df.columns:
        df["author"].to_frame().to_parquet(
            os.path.join(ACTIVATIONS_DIR, "authors.parquet"), index=False
        )

    # Guardar metadata
    meta = {
        "model": MODEL, "hookpoint": HOOKPOINT, "context_len": CONTEXT_LEN,
        "hidden_size": hidden_size, "n_comments": n,
        "trait": TRAIT_NAME, "num_classes": NUM_CLASSES,
    }
    with open(os.path.join(ACTIVATIONS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Activaciones guardadas en {ACTIVATIONS_DIR}/")


def cargar_o_extraer_activaciones(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    """Carga activaciones de disco si existen, sino las extrae y guarda.

    Returns: (last_token, mean_token, labels, authors_array, hidden_size)
    """
    meta_path = os.path.join(ACTIVATIONS_DIR, "meta.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get("n_comments") == len(df):
            print(f"Cargando activaciones desde {ACTIVATIONS_DIR}/ ...")
            last_token = np.load(os.path.join(ACTIVATIONS_DIR, "last_token.npy"), mmap_mode="r")
            mean_token = np.load(os.path.join(ACTIVATIONS_DIR, "mean_token.npy"), mmap_mode="r")
            labels = np.load(os.path.join(ACTIVATIONS_DIR, "labels.npy"))
            hidden_size = meta["hidden_size"]

            authors = None
            authors_path = os.path.join(ACTIVATIONS_DIR, "authors.parquet")
            if os.path.exists(authors_path):
                authors = pd.read_parquet(authors_path)["author"].to_numpy()

            print(f"  last_token: {last_token.shape}, mean_token: {mean_token.shape}")
            return last_token, mean_token, labels, authors, hidden_size
        else:
            print(f"Numero de comentarios cambio ({meta['n_comments']} -> {len(df)}). Re-extrayendo...")

    _extraer_y_guardar_activaciones(df)

    return cargar_o_extraer_activaciones(df)


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
    """Train/eval/test estratificado a nivel de usuario."""
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

        user_df = df[["author", "label"]].drop_duplicates("author")
        total_saved = len(train_auth) + len(eval_auth) + len(test_auth)
        if total_saved == len(user_df):
            print(f"Splits de usuarios cargados desde {SPLITS_DIR}/")
            return train_auth, eval_auth, test_auth
        print(f"Num usuarios cambio ({total_saved} -> {len(user_df)}). Regenerando...")

    user_df = df[["author", "label"]].drop_duplicates("author")
    authors = user_df["author"].to_numpy()
    user_labels = user_df["label"].to_numpy().astype(np.int8)

    train_eval_auth, test_auth = train_test_split(
        authors, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=user_labels,
    )
    # Recalcular labels para sub-split
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

    print(f"\n=== {nombre} ===")
    print(
        f"Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f} | "
        f"Precision macro: {prec_macro:.4f} | Recall macro: {rec_macro:.4f} | "
        f"F1 macro: {f1_mac:.4f} | F1 weighted: {f1_w:.4f}"
    )
    for i, cn in enumerate(CLASS_NAMES):
        print(f"  {cn:>8s}: prec={prec_c[i]:.4f} rec={rec_c[i]:.4f} f1={f1_c[i]:.4f}")
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
    print(f"ENTRENANDO: {run_name}")
    print(f"  Train: {len(y_train):,} | Eval: {len(y_eval):,}")
    if class_weights is not None:
        cw_str = ", ".join(f"{CLASS_NAMES[i]}={class_weights[i]:.3f}" for i in range(NUM_CLASSES))
        print(f"  Pesos de clase: {cw_str}")
    else:
        print(f"  Pesos de clase: ninguno (todos 1.0)")

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

    Escanea features por chunks secuenciales para evitar copiar grandes bloques
    del memmap a RAM.
    """
    num_features = features.shape[1]

    auth_list = sorted(author_set)
    auth_to_idx = {a: i for i, a in enumerate(auth_list)}
    n_users = len(auth_list)

    user_sums = np.zeros((n_users, num_features), dtype=np.float64)
    user_counts = np.zeros(n_users, dtype=np.int64)
    user_labels = np.full(n_users, -1, dtype=np.int64)

    row_to_user = np.full(len(authors), -1, dtype=np.int64)
    for i, auth in enumerate(authors):
        if auth in auth_to_idx:
            uidx = auth_to_idx[auth]
            row_to_user[i] = uidx
            if user_labels[uidx] == -1:
                user_labels[uidx] = labels[i]

    for start in range(0, len(authors), USER_AGG_CHUNK_SIZE):
        end = min(start + USER_AGG_CHUNK_SIZE, len(authors))
        chunk_user_ids = row_to_user[start:end]
        valid_mask = chunk_user_ids >= 0
        if not valid_mask.any():
            continue

        chunk_feats = np.asarray(features[start:end], dtype=np.float32)
        np.add.at(user_sums, chunk_user_ids[valid_mask], chunk_feats[valid_mask].astype(np.float64))
        np.add.at(user_counts, chunk_user_ids[valid_mask], 1)

    valid = user_counts > 0
    user_sums[valid] /= user_counts[valid, np.newaxis]
    return user_sums[valid].astype(np.float32), user_labels[valid]


# =====================
# MAIN
# =====================


def main():
    print(f"CLASIFICADOR {TRAIT_NAME.upper()} - ACTIVACIONES DIRECTAS GPT-2 (NO SAE)")

    # 1. Cargar datos
    df = cargar_datos()

    # Verificar que hay columna author
    if "author" not in df.columns:
        print("AVISO: No hay columna 'author', se omitiran las predicciones a nivel usuario.")
        has_author = False
    else:
        df["author"] = df["author"].astype(str).str.strip()
        has_author = True

    # 2. Extraer / cargar activaciones
    last_token, mean_token, labels, authors, hidden_size = cargar_o_extraer_activaciones(df)
    print(f"\nActivaciones: hidden_size={hidden_size}, comentarios={len(labels):,}")

    # Distribucion de labels
    for i, cn in enumerate(CLASS_NAMES):
        count = int((labels == i).sum())
        print(f"  {cn}: {count:,} ({100*count/len(labels):.1f}%)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}
    trained_runs = {}
    train_auth = eval_auth = test_auth = None

    # ==============================
    # A) NIVEL COMENTARIO
    # ==============================
    print("\n" + "#" * 70)
    print("# A) CLASIFICACION A NIVEL DE COMENTARIO")
    print("#" * 70)

    train_idx, eval_idx, test_idx = dividir_comentarios(labels, df, authors)
    y_train_c = labels[train_idx]
    y_eval_c = labels[eval_idx]

    print(f"\nSplit comentarios: train={len(train_idx):,} eval={len(eval_idx):,} test={len(test_idx):,}")
    for i, cn in enumerate(CLASS_NAMES):
        tr_n = int((y_train_c == i).sum())
        ev_n = int((y_eval_c == i).sum())
        print(f"  {cn}: train={tr_n:,} eval={ev_n:,}")

    # Pre-calcular pesos de clase manuales sobre el train set
    train_class_weights_manual = calcular_pesos_clase_manual(y_train_c)

    comment_features = {
        "last_token": (last_token, train_idx, eval_idx),
        "mean": (mean_token, train_idx, eval_idx),
    }

    for pooling_name in COMMENT_POOLINGS:
        feats, tr_idx, ev_idx = comment_features[pooling_name]

        # Fit scaler incremental en train para no materializar todo el split
        print(f"\n  Ajustando StandardScaler para {pooling_name} (incremental)...")
        scaler = StandardScaler()
        for sc_start in range(0, len(tr_idx), SCALER_BATCH_SIZE):
            sc_end = min(sc_start + SCALER_BATCH_SIZE, len(tr_idx))
            chunk = np.asarray(feats[tr_idx[sc_start:sc_end]], dtype=np.float32)
            scaler.partial_fit(chunk)
        print(f"    Scaler ajustado sobre {len(tr_idx):,} muestras")

        # --- Configuraciones de balanceo ---
        for bal_cfg in BALANCE_CONFIGS:
            if bal_cfg["name"] == "undersampling":
                # Submuestrear al tamano de la clase minoritaria (via indices)
                us_idx = random_undersample_idx(y_train_c)
                clf, metrics = entrenar_comentario(
                    feats=feats, train_idx=tr_idx[us_idx], eval_idx=ev_idx,
                    y_train=y_train_c[us_idx], y_eval=y_eval_c,
                    class_weights=None,
                    pooling_name=pooling_name, balance_name=bal_cfg["name"],
                    scaler=scaler,
                )
            else:
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
            trained_runs[run_key] = {
                "level": "comentario",
                "pooling": pooling_name,
                "clf": clf,
                "scaler": scaler,
            }

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

            # Fit scaler en train de usuarios
            u_scaler = StandardScaler()
            for sc_start in range(0, len(X_u_train), SCALER_BATCH_SIZE):
                sc_end = min(sc_start + SCALER_BATCH_SIZE, len(X_u_train))
                u_scaler.partial_fit(X_u_train[sc_start:sc_end])

            # --- Configuraciones de balanceo ---
            for bal_cfg in BALANCE_CONFIGS:
                if bal_cfg["name"] == "undersampling":
                    X_u_us, y_u_us = random_undersample(X_u_train, y_u_train)
                    X_tr_n = u_scaler.transform(X_u_us)
                    X_ev_n = u_scaler.transform(X_u_eval)
                    cw = None
                else:
                    if not bal_cfg["use_class_weights"]:
                        cw = None
                    else:
                        cw = train_class_weights_manual

                    X_tr_n = u_scaler.transform(X_u_train)
                    X_ev_n = u_scaler.transform(X_u_eval)
                    X_u_us, y_u_us = X_u_train, y_u_train

                run_name = f"usuario_{pooling_name}_{bal_cfg['name']}"
                print(f"ENTRENANDO: {run_name}")
                print(f"  Train users: {len(y_u_us):,} | Eval users: {len(y_u_eval):,}")
                if cw is not None:
                    cw_str = ", ".join(f"{CLASS_NAMES[i]}={cw[i]:.3f}" for i in range(NUM_CLASSES))
                    print(f"  Pesos de clase: {cw_str}")
                else:
                    print(f"  Pesos de clase: ninguno (todos 1.0)")

                clf = SGDClassifier(
                    loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
                    random_state=RANDOM_STATE, average=True,
                )
                classes = np.arange(NUM_CLASSES, dtype=np.int64)
                sw = sample_weights_from_class_weights(y_u_us, cw)
                clf.partial_fit(X_tr_n, y_u_us, classes=classes, sample_weight=sw)

                for epoch in range(1, TRAIN_EPOCHS):
                    perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(len(y_u_us))
                    sw_perm = sample_weights_from_class_weights(y_u_us[perm], cw)
                    clf.partial_fit(X_tr_n[perm], y_u_us[perm], sample_weight=sw_perm)

                y_pred = clf.predict(X_ev_n)
                metrics = evaluar(f"EVAL {run_name}", y_u_eval, y_pred)

                run_key = run_name
                all_results[run_key] = metrics
                trained_runs[run_key] = {
                    "level": "usuario",
                    "pooling": pooling_name,
                    "clf": clf,
                    "scaler": u_scaler,
                }


            del X_u_train, y_u_train, X_u_eval, y_u_eval, X_tr_n, X_ev_n
            gc.collect()

    # ==============================
    # RESUMEN FINAL
    # ==============================
    print("RESUMEN DE RESULTADOS (EVAL)")
    header_f1 = " ".join(f"{'F1_'+cn:>8s}" for cn in CLASS_NAMES)
    print(f"{'Config':<45} {'Acc':>6} {'BalAcc':>7} {'F1mac':>6} {header_f1}")
    print("-" * (45 + 6 + 7 + 6 + 8 * NUM_CLASSES + NUM_CLASSES + 3))
    for key, m in all_results.items():
        f1_vals = " ".join(f"{m.get(f'f1_{cn}', 0.0):8.4f}" for cn in CLASS_NAMES)
        print(f"{key:<45} {m['accuracy']:.4f} {m['balanced_accuracy']:.5f} "
              f"{m['f1_macro']:.4f} {f1_vals}")

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

    def _eval_test_for_run(run_name: str, artifact: Dict) -> Dict[str, float]:
        if artifact["level"] == "comentario":
            test_feats = last_token if artifact["pooling"] == "last_token" else mean_token
            y_test_ = labels[test_idx]
            test_preds = []
            for start in range(0, len(test_idx), SCALER_BATCH_SIZE):
                end = min(start + SCALER_BATCH_SIZE, len(test_idx))
                X_test_ = np.asarray(test_feats[test_idx[start:end]], dtype=np.float32)
                X_test_ = artifact["scaler"].transform(X_test_)
                test_preds.append(artifact["clf"].predict(X_test_))
            y_test_pred_ = np.concatenate(test_preds)
        else:
            test_feats = last_token if artifact["pooling"] == "mean_of_last" else mean_token
            X_u_test_, y_test_ = _agregar_por_usuario(authors, test_feats, labels, set(test_auth))
            X_u_test_ = artifact["scaler"].transform(X_u_test_)
            y_test_pred_ = artifact["clf"].predict(X_u_test_)
        return evaluar(f"TEST {run_name}", y_test_, y_test_pred_)

    best_per_level_test: Dict[str, Optional[Tuple[str, Dict[str, float]]]] = {
        "comentario": None,
        "usuario": None,
    }
    for level in ("comentario", "usuario"):
        entry = best_per_level.get(level)
        if entry is None:
            continue
        run_name_lvl, _ = entry
        artifact_lvl = trained_runs[run_name_lvl]
        m_test = _eval_test_for_run(run_name_lvl, artifact_lvl)
        best_per_level_test[level] = (run_name_lvl, m_test)

    print_best_per_level_test(best_per_level_test)

    best_test_metrics = (
        best_per_level_test.get(best_artifact["level"])[1]
        if best_per_level_test.get(best_artifact["level"]) is not None
        else _eval_test_for_run(best_run, best_artifact)
    )

    # Guardar resumen JSON
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
