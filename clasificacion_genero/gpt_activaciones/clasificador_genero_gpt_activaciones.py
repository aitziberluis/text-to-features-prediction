"""
Clasificador de genero usando activaciones directas de GPT-2 (NO SAE).

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
from preprocesamiento import preparar_dataset_para_sae

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

# Directorio donde se guardan las activaciones extraidas
ACTIVATIONS_DIR = "data/activaciones_gpt2_genero"

# Splits
TEST_SIZE = 0.15
EVAL_SIZE = 0.15
RANDOM_STATE = 42
SPLITS_DIR = "data/splits_genero_70_15_15"

# Entrenamiento
EXTRACT_BATCH_SIZE = 32
TRAIN_EPOCHS = 1
SGD_ALPHA = 1e-5

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
]

# Resampling: submuestra maxima para SMOTE/ADASYN (RAM limitada)
MAX_RESAMPLE_TRAIN = 500_000

# Tecnicas de resampling a probar
RESAMPLE_CONFIGS = [
    {"name": "SMOTE", "cls": SMOTE, "kwargs": {"random_state": RANDOM_STATE}},
    {"name": "ADASYN", "cls": ADASYN, "kwargs": {"random_state": RANDOM_STATE}},
]

# Output
OUTPUT_DIR = "modelos/genero_gpt_activaciones"



# CARGA DE DATOS
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


# EXTRACCION DE ACTIVACIONES

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
    labels = np.array([0 if g == "f" else 1 for g in df["gender_clean"]], dtype=np.int8)
    np.save(os.path.join(ACTIVATIONS_DIR, "labels.npy"), labels)

    if "author" in df.columns:
        df["author"].to_frame().to_parquet(
            os.path.join(ACTIVATIONS_DIR, "authors.parquet"), index=False
        )

    # Guardar metadata
    meta = {
        "model": MODEL, "hookpoint": HOOKPOINT, "context_len": CONTEXT_LEN,
        "hidden_size": hidden_size, "n_comments": n,
        "label_map": {"f": 0, "m": 1},
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
    # Recalcular labels para sub-split
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
    prec_c = precision_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    rec_c = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    f1_c = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)

    print(f"\n=== {nombre} ===")
    print(f"Accuracy: {acc:.4f} | Balanced Acc: {bal_acc:.4f} | F1 macro: {f1_mac:.4f} | F1 weighted: {f1_w:.4f}")
    print(f"  female: prec={prec_c[0]:.4f} rec={rec_c[0]:.4f} f1={f1_c[0]:.4f}")
    print(f"  male:   prec={prec_c[1]:.4f} rec={rec_c[1]:.4f} f1={f1_c[1]:.4f}")
    print(classification_report(y_true, y_pred, target_names=["female", "male"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "accuracy": float(acc), "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_mac), "f1_weighted": float(f1_w),
        "precision_female": float(prec_c[0]), "recall_female": float(rec_c[0]), "f1_female": float(f1_c[0]),
        "precision_male": float(prec_c[1]), "recall_male": float(rec_c[1]), "f1_male": float(f1_c[1]),
    }


# =====================
# ENTRENAMIENTO NIVEL COMENTARIO
# =====================


def entrenar_comentario(
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval: np.ndarray, y_eval: np.ndarray,
    female_w: float, male_w: float,
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
            idx = perm[start:start + batch_size]
            xb = np.asarray(X_train[idx], dtype=np.float32)
            if scaler is not None:
                xb = scaler.transform(xb)
            yb = y_train[idx]
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
    female_w: float, male_w: float,
    pooling_name: str, balance_name: str,
    scaler: StandardScaler = None,
) -> Tuple[SGDClassifier, Dict]:
    """Entrena SGD a nivel usuario (features ya agregadas) y evalua en eval."""
    run_name = f"usuario_{pooling_name}_{balance_name}"

    print(f"\nAgregando features por usuario para {run_name}...")
    X_train, y_train = _agregar_por_usuario(authors, features, labels, set(train_auth))
    X_eval, y_eval = _agregar_por_usuario(authors, features, labels, set(eval_auth))

    # Normalizar
    if scaler is not None:
        X_train = scaler.transform(X_train)
        X_eval = scaler.transform(X_eval)

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

    # Usuarios son pocos, se puede hacer en un solo fit
    clf.partial_fit(X_train, y_train, classes=classes, sample_weight=sw)

    # Para mas epocas, re-shuffle
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
    print("CLASIFICADOR GENERO - ACTIVACIONES DIRECTAS GPT-2 (NO SAE)")
    print("=" * 70)

    # 1. Cargar datos
    df = cargar_datos_genero()

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
    print(f"  Train dist: f={int((y_train_c==0).sum()):,} m={int((y_train_c==1).sum()):,}")
    print(f"  Eval  dist: f={int((y_eval_c==0).sum()):,} m={int((y_eval_c==1).sum()):,}")

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
            clf, metrics = entrenar_comentario(
                X_train=X_tr, y_train=y_train_c,
                X_eval=X_ev, y_eval=y_eval_c,
                female_w=bal_cfg["female_w"], male_w=bal_cfg["male_w"],
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
            n_f = int((y_train_c == 0).sum())
            n_m = int((y_train_c == 1).sum())
            # Mantener proporcion original pero limitando total
            ratio = n_f / (n_f + n_m)
            n_f_sub = min(int(MAX_RESAMPLE_TRAIN * ratio), n_f)
            n_m_sub = min(MAX_RESAMPLE_TRAIN - n_f_sub, n_m)
            sub_idx_f = rng.choice(np.where(y_train_c == 0)[0], size=n_f_sub, replace=False)
            sub_idx_m = rng.choice(np.where(y_train_c == 1)[0], size=n_m_sub, replace=False)
            sub_idx = np.concatenate([sub_idx_f, sub_idx_m])
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
            print(f"    Antes: f={int((y_tr_sub==0).sum()):,} m={int((y_tr_sub==1).sum()):,}")

            try:
                sampler = resample_cfg["cls"](**resample_cfg["kwargs"])
                X_resampled, y_resampled = sampler.fit_resample(X_tr_sub_norm, y_tr_sub)
                print(f"    Despues: f={int((y_resampled==0).sum()):,} m={int((y_resampled==1).sum()):,}")
            except Exception as e:
                print(f"    ERROR en {resample_name}: {e}")
                continue

            # Entrenar con datos ya normalizados (train ya normalizado, eval normalizar)
            X_ev_norm = scaler.transform(X_ev)
            clf, metrics = entrenar_comentario(
                X_train=X_resampled, y_train=y_resampled,
                X_eval=X_ev_norm, y_eval=y_eval_c,
                female_w=1.0, male_w=1.0,
                pooling_name=pooling_name, balance_name=resample_name,
                scaler=None,  # ya normalizado
            )

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
                X_tr_n = u_scaler.transform(X_u_train)
                X_ev_n = u_scaler.transform(X_u_eval)

                run_name = f"usuario_{pooling_name}_{bal_cfg['name']}"
                print(f"\n{'='*60}")
                print(f"ENTRENANDO: {run_name}")
                print(f"  Train users: {len(y_u_train):,} | Eval users: {len(y_u_eval):,}")
                print(f"  Pesos: female={bal_cfg['female_w']}, male={bal_cfg['male_w']}")
                print(f"{'='*60}")

                clf = SGDClassifier(
                    loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
                    random_state=RANDOM_STATE, average=True,
                )
                classes = np.array([0, 1], dtype=np.int64)
                sw = np.where(y_u_train == 0, bal_cfg["female_w"], bal_cfg["male_w"]).astype(np.float32)
                clf.partial_fit(X_tr_n, y_u_train, classes=classes, sample_weight=sw)

                for epoch in range(1, TRAIN_EPOCHS):
                    perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(len(y_u_train))
                    sw_perm = np.where(y_u_train[perm] == 0, bal_cfg["female_w"], bal_cfg["male_w"]).astype(np.float32)
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

            print(f"\n  Usuarios train: f={int((y_u_train==0).sum()):,} m={int((y_u_train==1).sum()):,}")

            for resample_cfg in RESAMPLE_CONFIGS:
                resample_name = resample_cfg["name"]
                print(f"\n  Aplicando {resample_name} a nivel usuario ({len(y_u_train):,} usuarios)...")

                try:
                    sampler = resample_cfg["cls"](**resample_cfg["kwargs"])
                    X_resampled, y_resampled = sampler.fit_resample(X_tr_n, y_u_train)
                    print(f"    Despues: f={int((y_resampled==0).sum()):,} m={int((y_resampled==1).sum()):,}")
                except Exception as e:
                    print(f"    ERROR en {resample_name}: {e}")
                    continue

                run_name = f"usuario_{pooling_name}_{resample_name}"
                print(f"\n{'='*60}")
                print(f"ENTRENANDO: {run_name}")
                print(f"{'='*60}")

                clf = SGDClassifier(
                    loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
                    random_state=RANDOM_STATE, average=True,
                )
                classes = np.array([0, 1], dtype=np.int64)
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
    print(f"{'Config':<50} {'Acc':>6} {'BalAcc':>7} {'F1mac':>6} {'F1_f':>6} {'F1_m':>6}")
    print("-" * 85)
    for key, m in all_results.items():
        print(f"{key:<50} {m['accuracy']:.4f} {m['balanced_accuracy']:.5f} "
              f"{m['f1_macro']:.4f} {m['f1_female']:.4f} {m['f1_male']:.4f}")

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
