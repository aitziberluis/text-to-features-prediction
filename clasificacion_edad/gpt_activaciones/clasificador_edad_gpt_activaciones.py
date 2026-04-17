"""
Clasificador de edad usando activaciones directas de GPT-2 (NO SAE).

Extrae las activaciones de una capa intermedia de GPT-2, las guarda a disco
para no repetir el forward pass, y entrena clasificadores lineales (SGD)
en multiples configuraciones:

A nivel de COMENTARIO:
  - last_token: activacion del ultimo token real del comentario
  - mean: media de activaciones de todos los tokens reales
  - Cada uno con y sin balanceo de clase

A nivel de USUARIO:
  - mean_of_last: media de last_token de todos los comentarios del usuario
  - mean_of_mean: media de mean de todos los comentarios del usuario
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
from typing import Dict, List, Optional, Tuple

import dotenv
import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE, ADASYN
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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Importar desde el directorio raiz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocesamiento import preparar_dataset_para_edad

dotenv.load_dotenv()

# =====================
# CONFIGURACION
# =====================

MODEL = "openai-community/gpt2"
CONTEXT_LEN = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HOOKPOINT = "transformer.h.8"

# Rutas
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"
TEXT_COLUMN = "body"
MAX_COMMENTS = None

# Rangos de edad
AGE_GROUPS = ["14_19", "20_29", "30_39", "40_plus"]
LABEL_MAP = {g: i for i, g in enumerate(AGE_GROUPS)}
NUM_CLASSES = len(AGE_GROUPS)

# Directorio donde se guardan las activaciones extraidas
ACTIVATIONS_DIR = "data/activaciones_gpt2_edad"

# Directorio compartido para los indices de split (comun a GPT y SAE)
SPLITS_DIR = "data/splits_edad"

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
]

# Resampling: submuestra maxima para SMOTE/ADASYN (RAM limitada)
MAX_RESAMPLE_TRAIN = 500_000

# Tecnicas de resampling a probar
RESAMPLE_CONFIGS = [
    {"name": "SMOTE", "cls": SMOTE, "kwargs": {"random_state": RANDOM_STATE}},
    {"name": "ADASYN", "cls": ADASYN, "kwargs": {"random_state": RANDOM_STATE}},
]

# Output
OUTPUT_DIR = "modelos/edad_gpt_activaciones"


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
    labels = np.array([LABEL_MAP[g] for g in df["age_group"]], dtype=np.int8)
    np.save(os.path.join(ACTIVATIONS_DIR, "labels.npy"), labels)

    if "author" in df.columns:
        df["author"].to_frame().to_parquet(
            os.path.join(ACTIVATIONS_DIR, "authors.parquet"), index=False
        )

    # Guardar metadata
    meta = {
        "model": MODEL, "hookpoint": HOOKPOINT, "context_len": CONTEXT_LEN,
        "hidden_size": hidden_size, "n_comments": n,
        "label_map": LABEL_MAP,
        "age_groups": AGE_GROUPS,
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
    """Train/eval/test estratificado a nivel de usuario.

    Guarda los autores de cada split en SPLITS_DIR para reproducibilidad.
    Si ya existen y coinciden, los carga de disco.
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
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval: np.ndarray, y_eval: np.ndarray,
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
            idx = perm[start:start + batch_size]
            xb = np.asarray(X_train[idx], dtype=np.float32)
            if scaler is not None:
                xb = scaler.transform(xb)
            yb = y_train[idx]
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

    # Eval
    X_ev = np.asarray(X_eval, dtype=np.float32)
    if scaler is not None:
        X_ev = scaler.transform(X_ev)
    y_pred = clf.predict(X_ev)
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
    """Agrega features por media de usuario para un subconjunto de autores."""
    mask = np.isin(authors, list(author_set))
    sub_authors = authors[mask]
    sub_features = features[mask]
    sub_labels = labels[mask]

    # Agrupar con pandas para mayor velocidad
    df_tmp = pd.DataFrame({"author": sub_authors, "label": sub_labels})
    df_tmp["row"] = np.arange(len(sub_authors))

    user_feats = []
    user_labels = []

    for author, group in df_tmp.groupby("author", sort=False):
        rows = group["row"].to_numpy()
        user_feat = np.asarray(sub_features[rows]).mean(axis=0)
        user_feats.append(user_feat)
        user_labels.append(int(group["label"].iloc[0]))

    return np.array(user_feats, dtype=np.float32), np.array(user_labels, dtype=np.int64)


def entrenar_usuario(
    authors: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    train_auth: np.ndarray, eval_auth: np.ndarray,
    class_weights: Optional[np.ndarray],
    pooling_name: str, balance_name: str,
) -> Tuple[SGDClassifier, Dict]:
    """Entrena SGD a nivel usuario (features ya agregadas) y evalua en eval."""
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

    # Usuarios son pocos, se puede hacer en un solo fit
    clf.partial_fit(X_train, y_train, classes=classes, sample_weight=sw)

    # Para mas epocas, re-shuffle
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
    print("=" * 70)
    print("CLASIFICADOR EDAD - ACTIVACIONES DIRECTAS GPT-2 (NO SAE)")
    print("=" * 70)

    # 1. Cargar datos
    df = cargar_datos_edad()

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
    for i, group in enumerate(AGE_GROUPS):
        count = int((labels == i).sum())
        print(f"  {group}: {count:,} ({100*count/len(labels):.1f}%)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

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
        X_tr = np.asarray(feats[tr_idx], dtype=np.float32)
        X_ev = np.asarray(feats[ev_idx], dtype=np.float32)

        # Fit scaler en train
        print(f"\n  Ajustando StandardScaler para {pooling_name}...")
        scaler = StandardScaler()
        scaler.fit(X_tr)

        # --- Configuraciones con pesos de clase (sin resampling) ---
        for bal_cfg in BALANCE_CONFIGS:
            if not bal_cfg["use_class_weights"]:
                cw = None
            else:
                cw = train_class_weights_manual

            clf, metrics = entrenar_comentario(
                X_train=X_tr, y_train=y_train_c,
                X_eval=X_ev, y_eval=y_eval_c,
                class_weights=cw,
                pooling_name=pooling_name, balance_name=bal_cfg["name"],
                scaler=scaler,
            )

            run_key = f"comentario_{pooling_name}_{bal_cfg['name']}"
            all_results[run_key] = metrics

            model_path = os.path.join(OUTPUT_DIR, f"{run_key}.pkl")
            joblib.dump(clf, model_path)
            print(f"  -> Modelo guardado: {model_path}")

        # --- Configuraciones con resampling (SMOTE, ADASYN) ---
        # Subsamplear si es necesario para caber en RAM
        n_tr = len(y_train_c)
        if n_tr > MAX_RESAMPLE_TRAIN:
            print(f"\n  Submuestreando {n_tr:,} -> {MAX_RESAMPLE_TRAIN:,} para resampling...")
            rng = np.random.RandomState(RANDOM_STATE)
            # Mantener proporcion original pero limitando total
            counts = np.bincount(y_train_c, minlength=NUM_CLASSES)
            fracs = counts / counts.sum()
            sub_counts = np.round(fracs * MAX_RESAMPLE_TRAIN).astype(int)
            sub_idx_parts = []
            for c in range(NUM_CLASSES):
                c_idx = np.where(y_train_c == c)[0]
                n_take = min(sub_counts[c], len(c_idx))
                sub_idx_parts.append(rng.choice(c_idx, size=n_take, replace=False))
            sub_idx = np.concatenate(sub_idx_parts)
            X_tr_sub = X_tr[sub_idx]
            y_tr_sub = y_train_c[sub_idx]
        else:
            X_tr_sub = X_tr
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

            # Entrenar con datos ya normalizados (train ya normalizado, eval normalizar)
            X_ev_norm = scaler.transform(X_ev)
            clf, metrics = entrenar_comentario(
                X_train=X_resampled, y_train=y_resampled,
                X_eval=X_ev_norm, y_eval=y_eval_c,
                class_weights=None,
                pooling_name=pooling_name, balance_name=resample_name,
                scaler=None,  # ya normalizado
            )

            # Continuar entrenamiento sobre TODOS los datos reales
            print(f"  Entrenando sobre todos los datos reales ({n_tr:,} muestras)...")
            batch_size_all = 4096
            total_steps_all = math.ceil(n_tr / batch_size_all)
            last_print_t = time.time()
            perm_all = np.random.RandomState(RANDOM_STATE + TRAIN_EPOCHS).permutation(n_tr)
            for step, start_r in enumerate(range(0, n_tr, batch_size_all)):
                idx_a = perm_all[start_r:start_r + batch_size_all]
                xb = scaler.transform(X_tr[idx_a])
                yb_a = y_train_c[idx_a]
                clf.partial_fit(xb, yb_a)

                now = time.time()
                if now - last_print_t >= PROGRESS_INTERVAL or step == total_steps_all - 1:
                    pct = 100.0 * (step + 1) / total_steps_all
                    print(f"  [All data] {pct:5.1f}% ({step+1}/{total_steps_all})")
                    last_print_t = now

            # Re-evaluar tras entrenamiento completo
            y_pred = clf.predict(X_ev_norm)
            metrics = evaluar(f"EVAL comentario_{pooling_name}_{resample_name}", y_eval_c, y_pred)

            run_key = f"comentario_{pooling_name}_{resample_name}"
            all_results[run_key] = metrics

            model_path = os.path.join(OUTPUT_DIR, f"{run_key}.pkl")
            joblib.dump(clf, model_path)
            print(f"  -> Modelo guardado: {model_path}")

    # ==============================
    # B) NIVEL USUARIO
    # ==============================
    if has_author and authors is not None:
        print("\n" + "#" * 70)
        print("# B) CLASIFICACION A NIVEL DE USUARIO")
        print("#" * 70)

        train_auth, eval_auth, test_auth = dividir_usuarios(df)
        print(f"\nSplit usuarios: train={len(train_auth):,} eval={len(eval_auth):,} test={len(test_auth):,}")

        # mean_of_last: para cada usuario, media de last_token de sus comentarios
        # mean_of_mean: para cada usuario, media de mean_token de sus comentarios
        user_features = {
            "mean_of_last": last_token,
            "mean_of_mean": mean_token,
        }

        for pooling_name in USER_POOLINGS:
            feats = user_features[pooling_name]

            # Pre-agregar para poder hacer scaler y resampling
            print(f"\n  Agregando features por usuario para {pooling_name}...")
            X_u_train, y_u_train = _agregar_por_usuario(authors, feats, labels, set(train_auth))
            X_u_eval, y_u_eval = _agregar_por_usuario(authors, feats, labels, set(eval_auth))

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

                run_key = run_name
                all_results[run_key] = metrics

                model_path = os.path.join(OUTPUT_DIR, f"{run_key}.pkl")
                joblib.dump(clf, model_path)
                print(f"  -> Modelo guardado: {model_path}")

            # --- Configuraciones con resampling (SMOTE, ADASYN) ---
            X_tr_n = u_scaler.transform(X_u_train)
            X_ev_n = u_scaler.transform(X_u_eval)

            print(f"\n  Usuarios train dist:")
            for i, group in enumerate(AGE_GROUPS):
                print(f"    {group}: {int((y_u_train==i).sum()):,}")

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

                model_path = os.path.join(OUTPUT_DIR, f"{run_name}.pkl")
                joblib.dump(clf, model_path)
                print(f"  -> Modelo guardado: {model_path}")

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

    # Guardar resumen JSON
    summary_path = os.path.join(OUTPUT_DIR, "resultados_resumen.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResumen guardado en: {summary_path}")

    print("\n" + "=" * 70)
    print("COMPLETADO - Test reservado para uso futuro")
    print("=" * 70)


if __name__ == "__main__":
    main()
