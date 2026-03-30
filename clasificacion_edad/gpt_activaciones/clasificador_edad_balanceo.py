"""
Clasificador de edad con tecnicas de BALANCEO PRE-ENTRENAMIENTO.

Carga activaciones pre-extraidas de data/activaciones_gpt2_edad/
y aplica distintas tecnicas de resampling ANTES del entrenamiento:

  1. Random Oversampling (ROS): duplica ejemplos de clases minoritarias
  2. SMOTE: genera ejemplos sinteticos interpolando entre vecinos
  3. ADASYN: como SMOTE pero genera mas ejemplos en zonas dificiles
  4. Random Undersampling (RUS): elimina ejemplos de la clase mayoritaria
  5. SMOTE + Tomek Links (combinado): SMOTE + limpieza de ejemplos ambiguos
  6. SMOTE + ENN (combinado): SMOTE + limpieza mas agresiva con ENN

Cada tecnica se prueba con:
  - MLP (PyTorch, 768->256->64->4) + CrossEntropy
  - SGD lineal normalizado

Tanto a nivel de COMENTARIO como de USUARIO.

StandardScaler se aplica ANTES del resampling para que SMOTE opere
en el espacio normalizado.

Rangos de edad: 14_19, 20_29, 30_39, 40_plus
Evaluacion solo en eval set (test reservado).
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
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
from sklearn.preprocessing import StandardScaler

# =====================
# CONFIGURACION
# =====================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ACTIVATIONS_DIR = "data/activaciones_gpt2_edad"
SPLITS_DIR = "data/splits_edad"

AGE_GROUPS = ["14_19", "20_29", "30_39", "40_plus"]
LABEL_MAP = {g: i for i, g in enumerate(AGE_GROUPS)}
NUM_CLASSES = len(AGE_GROUPS)
RANDOM_STATE = 42

# SGD
SGD_EPOCHS = 5
SGD_ALPHA = 1e-5
SGD_BATCH_SIZE = 4096

# MLP
MLP_HIDDEN = [256, 64]
MLP_DROPOUT = 0.3
MLP_LR = 1e-3
MLP_EPOCHS = 8
MLP_BATCH_SIZE = 4096
MLP_WEIGHT_DECAY = 1e-4

# Progreso
PROGRESS_INTERVAL = 3600

# Poolings
COMMENT_POOLINGS = ["last_token", "mean"]
USER_POOLINGS = ["mean_of_last", "mean_of_mean"]

# Para SMOTE: el dataset es enorme (5.5M). Resamplear todo es inviable
# en RAM, asi que se usa un subconjunto de train para resamplear.
# None = usar todo (solo viable si hay RAM suficiente).
MAX_TRAIN_RESAMPLE = 500_000


# =====================
# TECNICAS DE RESAMPLING
# =====================


def get_resamplers() -> list:
    """Devuelve lista de (nombre, resampler) a probar."""
    return [
        ("sin_balanceo", None),
        ("random_oversample", RandomOverSampler(random_state=RANDOM_STATE)),
        ("random_undersample", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("adasyn", ADASYN(random_state=RANDOM_STATE)),
        ("smote_tomek", SMOTETomek(random_state=RANDOM_STATE)),
        ("smote_enn", SMOTEENN(random_state=RANDOM_STATE)),
    ]


def aplicar_resampling(
    X: np.ndarray, y: np.ndarray, nombre: str, resampler
) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica resampling y muestra distribucion resultante."""
    if resampler is None:
        return X, y

    print(f"\n  Aplicando {nombre}...")
    print(f"    Antes: {np.bincount(y, minlength=NUM_CLASSES).tolist()} (total: {len(y):,})")

    t0 = time.time()
    X_res, y_res = resampler.fit_resample(X, y)
    elapsed = time.time() - t0

    print(f"    Despues: {np.bincount(y_res, minlength=NUM_CLASSES).tolist()} (total: {len(y_res):,})")
    print(f"    Tiempo: {elapsed:.1f}s")
    return X_res, y_res


def subsample_train(X: np.ndarray, y: np.ndarray, max_n: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Subsamplea el train set si es demasiado grande para SMOTE."""
    if max_n is None or len(y) <= max_n:
        return X, y

    print(f"  Submuestreando train: {len(y):,} -> {max_n:,} (para que SMOTE sea viable)")
    rng = np.random.RandomState(RANDOM_STATE)
    # Stratified subsample
    idx_all = []
    per_class = max_n // NUM_CLASSES
    for c in range(NUM_CLASSES):
        c_idx = np.where(y == c)[0]
        n_take = min(per_class, len(c_idx))
        chosen = rng.choice(c_idx, size=n_take, replace=False)
        idx_all.append(chosen)
    idx = np.concatenate(idx_all)
    rng.shuffle(idx)
    return X[idx], y[idx]


# =====================
# MLP MODEL
# =====================


class AgeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================
# CARGA DE DATOS
# =====================


def cargar_activaciones() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    meta_path = os.path.join(ACTIVATIONS_DIR, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No se encuentran activaciones en {ACTIVATIONS_DIR}/. "
            "Primero ejecuta clasificador_edad_gpt_activaciones.py."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    hidden_size = meta["hidden_size"]
    print(f"Cargando activaciones de {ACTIVATIONS_DIR}/ ...")
    last_token = np.load(os.path.join(ACTIVATIONS_DIR, "last_token.npy"), mmap_mode="r")
    mean_token = np.load(os.path.join(ACTIVATIONS_DIR, "mean_token.npy"), mmap_mode="r")
    labels = np.load(os.path.join(ACTIVATIONS_DIR, "labels.npy"))

    authors = None
    authors_path = os.path.join(ACTIVATIONS_DIR, "authors.parquet")
    if os.path.exists(authors_path):
        authors = pd.read_parquet(authors_path)["author"].to_numpy()

    print(f"  last_token: {last_token.shape}, mean_token: {mean_token.shape}")
    return last_token, mean_token, labels, authors, hidden_size


def cargar_splits(n_total: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_path = os.path.join(SPLITS_DIR, "split_comentarios_por_usuario.npz")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"No se encuentra split en {split_path}")
    data = np.load(split_path)
    return data["train_idx"], data["eval_idx"], data["test_idx"]


def cargar_splits_usuarios() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = {s: os.path.join(SPLITS_DIR, f"usuario_{s}_auth.npy") for s in ("train", "eval", "test")}
    return (
        np.load(paths["train"], allow_pickle=True),
        np.load(paths["eval"], allow_pickle=True),
        np.load(paths["test"], allow_pickle=True),
    )


# =====================
# NORMALIZACION INCREMENTAL
# =====================


def fit_scaler_incremental(X: np.ndarray, indices: np.ndarray, batch_size: int = 50000) -> StandardScaler:
    scaler = StandardScaler()
    n = len(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = np.asarray(X[indices[start:end]], dtype=np.float32)
        scaler.partial_fit(batch)
    return scaler


def transform_in_batches(X: np.ndarray, indices: np.ndarray, scaler: StandardScaler,
                         batch_size: int = 100000) -> np.ndarray:
    n = len(indices)
    dim = X.shape[1]
    result = np.empty((n, dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = np.asarray(X[indices[start:end]], dtype=np.float32)
        result[start:end] = scaler.transform(batch)
    return result


# =====================
# EVALUACION
# =====================


def evaluar(nombre: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
# ENTRENAMIENTO SGD
# =====================


def entrenar_sgd(X_train: np.ndarray, y_train: np.ndarray,
                 X_eval: np.ndarray, y_eval: np.ndarray,
                 run_name: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"SGD: {run_name}")
    print(f"  Train: {len(y_train):,} | Eval: {len(y_eval):,} | Epocas: {SGD_EPOCHS}")
    print(f"{'='*60}")

    clf = SGDClassifier(
        loss="log_loss", alpha=SGD_ALPHA, max_iter=1, tol=None,
        random_state=RANDOM_STATE, average=True,
    )
    classes = np.arange(NUM_CLASSES, dtype=np.int64)
    n = len(y_train)
    total_steps = math.ceil(n / SGD_BATCH_SIZE)
    last_print = time.time()

    for epoch in range(SGD_EPOCHS):
        perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(n)
        for step, start in enumerate(range(0, n, SGD_BATCH_SIZE)):
            idx = perm[start:start + SGD_BATCH_SIZE]
            xb = X_train[idx]
            yb = y_train[idx]
            if epoch == 0 and step == 0:
                clf.partial_fit(xb, yb, classes=classes)
            else:
                clf.partial_fit(xb, yb)

            now = time.time()
            if now - last_print >= PROGRESS_INTERVAL or (epoch == SGD_EPOCHS - 1 and step == total_steps - 1):
                pct = 100.0 * (epoch * total_steps + step + 1) / (SGD_EPOCHS * total_steps)
                print(f"  [{pct:5.1f}%] Epoch {epoch+1}/{SGD_EPOCHS} step {step+1}/{total_steps}")
                last_print = now

    y_pred = clf.predict(X_eval)
    return evaluar(f"EVAL {run_name}", y_eval, y_pred)


# =====================
# ENTRENAMIENTO MLP
# =====================


def entrenar_mlp(X_train: np.ndarray, y_train: np.ndarray,
                 X_eval: np.ndarray, y_eval: np.ndarray,
                 run_name: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"MLP: {run_name}")
    print(f"  Train: {len(y_train):,} | Eval: {len(y_eval):,}")
    print(f"  Hidden: {MLP_HIDDEN} | Dropout: {MLP_DROPOUT} | Epochs: {MLP_EPOCHS}")
    print(f"{'='*60}")

    input_dim = X_train.shape[1]
    model = AgeMLP(input_dim, MLP_HIDDEN, NUM_CLASSES, MLP_DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)

    n_train = len(y_train)
    steps_per_epoch = math.ceil(n_train / MLP_BATCH_SIZE)
    total_steps = MLP_EPOCHS * steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    last_print = time.time()

    for epoch in range(MLP_EPOCHS):
        model.train()
        perm = np.random.RandomState(RANDOM_STATE + epoch).permutation(n_train)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for start in range(0, n_train, MLP_BATCH_SIZE):
            end = min(start + MLP_BATCH_SIZE, n_train)
            idx = perm[start:end]
            xb = torch.tensor(X_train[idx], dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(y_train[idx], dtype=torch.long, device=DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * len(yb)
            epoch_correct += (logits.argmax(dim=1) == yb).sum().item()
            epoch_total += len(yb)

        avg_loss = epoch_loss / epoch_total
        avg_acc = epoch_correct / epoch_total
        now = time.time()
        if now - last_print >= PROGRESS_INTERVAL or epoch == 0 or epoch == MLP_EPOCHS - 1:
            print(f"  Epoch {epoch+1}/{MLP_EPOCHS}: loss={avg_loss:.4f} train_acc={avg_acc:.4f}")
            last_print = now

    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(y_eval), MLP_BATCH_SIZE):
            end = min(start + MLP_BATCH_SIZE, len(y_eval))
            xb = torch.tensor(X_eval[start:end], dtype=torch.float32, device=DEVICE)
            all_preds.append(xb.argmax(dim=1).cpu().numpy() if False else model(xb).argmax(dim=1).cpu().numpy())

    y_pred = np.concatenate(all_preds)
    return evaluar(f"EVAL {run_name}", y_eval, y_pred)


# =====================
# AGREGACION POR USUARIO
# =====================


def agregar_por_usuario(
    authors: np.ndarray, features: np.ndarray,
    labels: np.ndarray, author_set: set,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isin(authors, list(author_set))
    sub_authors = authors[mask]
    sub_features = features[mask]
    sub_labels = labels[mask]

    df_tmp = pd.DataFrame({"author": sub_authors, "label": sub_labels})
    df_tmp["row"] = np.arange(len(sub_authors))

    user_feats = []
    user_labels = []
    for author, group in df_tmp.groupby("author", sort=False):
        rows = group["row"].to_numpy()
        user_feats.append(np.asarray(sub_features[rows]).mean(axis=0))
        user_labels.append(int(group["label"].iloc[0]))

    return np.array(user_feats, dtype=np.float32), np.array(user_labels, dtype=np.int64)


# =====================
# MAIN
# =====================


def main():
    print("=" * 70)
    print("CLASIFICADOR EDAD - TECNICAS DE BALANCEO PRE-ENTRENAMIENTO")
    print("=" * 70)

    # 1. Cargar datos
    last_token, mean_token, labels, authors, hidden_size = cargar_activaciones()
    has_author = authors is not None

    print(f"\nDistribucion ({len(labels):,} comentarios):")
    for i, g in enumerate(AGE_GROUPS):
        c = int((labels == i).sum())
        print(f"  {g}: {c:,} ({100*c/len(labels):.1f}%)")

    # 2. Splits
    train_idx, eval_idx, test_idx = cargar_splits(len(labels))
    y_train_c = labels[train_idx]
    y_eval_c = labels[eval_idx]
    print(f"\nSplit: train={len(train_idx):,} eval={len(eval_idx):,} test={len(test_idx):,}")

    all_results = {}
    resamplers = get_resamplers()

    # ==============================
    # A) NIVEL COMENTARIO
    # ==============================
    feature_map = {"last_token": last_token, "mean": mean_token}

    for pooling_name in COMMENT_POOLINGS:
        raw_feats = feature_map[pooling_name]

        print(f"\n{'#'*70}")
        print(f"# COMENTARIO - pooling: {pooling_name}")
        print(f"{'#'*70}")

        # Normalizar
        print(f"\nFitting StandardScaler ({pooling_name})...")
        scaler = fit_scaler_incremental(raw_feats, train_idx)
        X_train_norm = transform_in_batches(raw_feats, train_idx, scaler)
        X_eval_norm = transform_in_batches(raw_feats, eval_idx, scaler)

        for resample_name, resampler in resamplers:
            print(f"\n{'─'*60}")
            print(f"RESAMPLING: {resample_name} (pooling={pooling_name})")
            print(f"{'─'*60}")

            if resampler is not None:
                # Subsamplear si es necesario para que SMOTE sea viable
                X_sub, y_sub = subsample_train(X_train_norm, y_train_c, MAX_TRAIN_RESAMPLE)
                try:
                    X_res, y_res = aplicar_resampling(X_sub, y_sub, resample_name, resampler)
                except Exception as e:
                    print(f"  ERROR en {resample_name}: {e}")
                    continue
            else:
                # Sin balanceo: usar subconjunto tambien para comparacion justa
                X_res, y_res = subsample_train(X_train_norm, y_train_c, MAX_TRAIN_RESAMPLE)

            # SGD
            key = f"comentario_{pooling_name}_{resample_name}_sgd"
            metrics = entrenar_sgd(X_res, y_res, X_eval_norm, y_eval_c, run_name=key)
            all_results[key] = metrics

            # MLP
            key = f"comentario_{pooling_name}_{resample_name}_mlp"
            metrics = entrenar_mlp(X_res, y_res, X_eval_norm, y_eval_c, run_name=key)
            all_results[key] = metrics

            # Liberar memoria del resampleo
            if resampler is not None:
                del X_res, y_res

        del X_train_norm, X_eval_norm

    # ==============================
    # B) NIVEL USUARIO
    # ==============================
    if has_author:
        print(f"\n{'#'*70}")
        print(f"# CLASIFICACION A NIVEL DE USUARIO")
        print(f"{'#'*70}")

        train_auth, eval_auth, test_auth = cargar_splits_usuarios()
        user_feature_map = {"mean_of_last": last_token, "mean_of_mean": mean_token}

        for pooling_name in USER_POOLINGS:
            raw_feats = user_feature_map[pooling_name]

            print(f"\n--- Usuario pooling: {pooling_name} ---")
            X_train_user, y_train_user = agregar_por_usuario(
                authors, raw_feats, labels, set(train_auth),
            )
            X_eval_user, y_eval_user = agregar_por_usuario(
                authors, raw_feats, labels, set(eval_auth),
            )

            # Normalizar
            scaler_u = StandardScaler()
            X_train_user = scaler_u.fit_transform(X_train_user)
            X_eval_user = scaler_u.transform(X_eval_user)

            print(f"  Usuarios: train={len(y_train_user):,} eval={len(y_eval_user):,}")
            print(f"  Dist train: {np.bincount(y_train_user, minlength=NUM_CLASSES).tolist()}")

            for resample_name, resampler in resamplers:
                print(f"\n{'─'*60}")
                print(f"RESAMPLING USUARIO: {resample_name} (pooling={pooling_name})")
                print(f"{'─'*60}")

                try:
                    X_res, y_res = aplicar_resampling(
                        X_train_user, y_train_user, resample_name, resampler,
                    )
                except Exception as e:
                    print(f"  ERROR en {resample_name}: {e}")
                    continue

                # SGD
                key = f"usuario_{pooling_name}_{resample_name}_sgd"
                metrics = entrenar_sgd(X_res, y_res, X_eval_user, y_eval_user, run_name=key)
                all_results[key] = metrics

                # MLP
                key = f"usuario_{pooling_name}_{resample_name}_mlp"
                metrics = entrenar_mlp(X_res, y_res, X_eval_user, y_eval_user, run_name=key)
                all_results[key] = metrics

            del X_train_user, X_eval_user

    # ==============================
    # RESUMEN FINAL
    # ==============================
    print("\n\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS (EVAL)")
    print("=" * 70)
    header_f1 = " ".join(f"{'F1_'+g:>8s}" for g in AGE_GROUPS)
    print(f"{'Config':<55} {'Acc':>6} {'BalAcc':>7} {'F1mac':>6} {header_f1}")
    print("-" * (55 + 6 + 7 + 6 + 8 * NUM_CLASSES + NUM_CLASSES + 3))
    for key, m in all_results.items():
        f1_vals = " ".join(f"{m.get(f'f1_{g}', 0.0):8.4f}" for g in AGE_GROUPS)
        print(f"{key:<55} {m['accuracy']:.4f} {m['balanced_accuracy']:.5f} "
              f"{m['f1_macro']:.4f} {f1_vals}")

    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
