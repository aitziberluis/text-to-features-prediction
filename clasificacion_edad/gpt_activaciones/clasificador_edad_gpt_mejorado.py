"""
Clasificador de edad MEJORADO usando activaciones de GPT-2.

Carga las activaciones ya extraidas en data/activaciones_gpt2_edad/
y los splits de data/splits_edad/, y entrena clasificadores mejorados:

Mejoras respecto al baseline (clasificador_edad_gpt_activaciones.py):
  1. StandardScaler: normalización de features (critico para modelos lineales)
  2. MLP no-lineal (PyTorch): captura patrones no-lineales que SGD lineal no puede
  3. Focal Loss: manejo de desbalanceo mucho mejor que sample weights
  4. Más épocas de entrenamiento con learning rate scheduling
  5. SGD mejorado: con normalización y mas épocas como baseline mejorado

Configuraciones:
  A) COMENTARIO:
     - SGD lineal + StandardScaler (5 épocas)       [sin/con balanceo]
     - MLP PyTorch + CrossEntropy + class weights    [sin/con balanceo]
     - MLP PyTorch + Focal Loss                      [gamma=1, gamma=2]
  B) USUARIO:
     - Mismos clasificadores sobre features agregadas por usuario

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

# Importar desde el directorio raiz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocesamiento import preparar_dataset_para_edad

# =====================
# CONFIGURACION
# =====================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Rutas de datos pre-extraidos
ACTIVATIONS_DIR = "data/activaciones_gpt2_edad"
SPLITS_DIR = "data/splits_edad"
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"

# Rangos de edad
AGE_GROUPS = ["14_19", "20_29", "30_39", "40_plus"]
LABEL_MAP = {g: i for i, g in enumerate(AGE_GROUPS)}
NUM_CLASSES = len(AGE_GROUPS)
RANDOM_STATE = 42

# SGD mejorado
SGD_EPOCHS = 5
SGD_ALPHA = 1e-5
SGD_BATCH_SIZE = 4096

# MLP PyTorch
MLP_HIDDEN = [256, 64]
MLP_DROPOUT = 0.3
MLP_LR = 1e-3
MLP_EPOCHS = 8
MLP_BATCH_SIZE = 4096
MLP_WEIGHT_DECAY = 1e-4

# Focal Loss gammas a probar
FOCAL_GAMMAS = [1.0, 2.0]

# Progreso
PROGRESS_INTERVAL = 3600

# Poolings
COMMENT_POOLINGS = ["last_token", "mean"]
USER_POOLINGS = ["mean_of_last", "mean_of_mean"]


# =====================
# FOCAL LOSS
# =====================


class FocalLoss(nn.Module):
    """Focal Loss para clasificacion multiclase desbalanceada.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduce la contribucion de ejemplos faciles (los de la clase mayoritaria
    que se clasifican bien) y se centra en los dificiles.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce_loss).mean()


# =====================
# MLP MODEL
# =====================


class AgeMLP(nn.Module):
    """MLP simple para clasificacion de edad."""

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
    """Carga activaciones pre-extraidas de disco."""
    meta_path = os.path.join(ACTIVATIONS_DIR, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No se encuentran activaciones en {ACTIVATIONS_DIR}/. "
            "Primero ejecuta clasificador_edad_gpt_activaciones.py para extraerlas."
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
    print(f"  labels: {labels.shape}, hidden_size: {hidden_size}")
    return last_token, mean_token, labels, authors, hidden_size


def cargar_splits(n_total: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga split de comentarios por usuario."""
    split_path = os.path.join(SPLITS_DIR, "split_comentarios_por_usuario.npz")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"No se encuentra split en {split_path}. "
            "Primero ejecuta clasificador_edad_gpt_activaciones.py."
        )
    data = np.load(split_path)
    train_idx = data["train_idx"]
    eval_idx = data["eval_idx"]
    test_idx = data["test_idx"]
    print(f"Split cargado: train={len(train_idx):,} eval={len(eval_idx):,} test={len(test_idx):,}")
    return train_idx, eval_idx, test_idx


def cargar_splits_usuarios() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga split de usuarios."""
    paths = {
        s: os.path.join(SPLITS_DIR, f"usuario_{s}_auth.npy")
        for s in ("train", "eval", "test")
    }
    for s, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"No se encuentra {p}")

    train_auth = np.load(paths["train"], allow_pickle=True)
    eval_auth = np.load(paths["eval"], allow_pickle=True)
    test_auth = np.load(paths["test"], allow_pickle=True)
    print(f"Split usuarios: train={len(train_auth):,} eval={len(eval_auth):,} test={len(test_auth):,}")
    return train_auth, eval_auth, test_auth


# =====================
# NORMALIZACION INCREMENTAL
# =====================


def fit_scaler_incremental(X: np.ndarray, indices: np.ndarray, batch_size: int = 50000) -> StandardScaler:
    """Fit StandardScaler incrementalmente sobre un subconjunto de X (memmap-compatible)."""
    scaler = StandardScaler()
    n = len(indices)
    total_steps = math.ceil(n / batch_size)
    last_print = time.time()

    for step, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        batch = np.asarray(X[batch_idx], dtype=np.float32)
        scaler.partial_fit(batch)

        now = time.time()
        if now - last_print >= PROGRESS_INTERVAL or step == total_steps - 1:
            pct = 100.0 * (step + 1) / total_steps
            print(f"  Scaler fit: {pct:5.1f}% ({end:,}/{n:,})")
            last_print = now

    return scaler


def transform_in_batches(X: np.ndarray, indices: np.ndarray, scaler: StandardScaler,
                         batch_size: int = 100000) -> np.ndarray:
    """Transforma un subconjunto de X con scaler, en batches para no saturar RAM."""
    n = len(indices)
    dim = X.shape[1]
    result = np.empty((n, dim), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        batch = np.asarray(X[batch_idx], dtype=np.float32)
        result[start:end] = scaler.transform(batch)

    return result


# =====================
# PESOS DE CLASE
# =====================


def calcular_pesos_clase(y: np.ndarray) -> np.ndarray:
    """Pesos inversos a la frecuencia (sklearn-style)."""
    counts = np.bincount(y, minlength=NUM_CLASSES)
    total = counts.sum()
    weights = np.where(counts > 0, total / (NUM_CLASSES * counts), 1.0)
    return weights.astype(np.float32)


def sample_weights_from_class_weights(y: np.ndarray, class_weights: Optional[np.ndarray]) -> np.ndarray:
    if class_weights is None:
        return np.ones(len(y), dtype=np.float32)
    return class_weights[y]


# =====================
# EVALUACION
# =====================


def evaluar(nombre: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Imprime y devuelve metricas de evaluacion multiclase."""
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
# SGD MEJORADO (con normalizacion)
# =====================


def entrenar_sgd_normalizado(
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval: np.ndarray, y_eval: np.ndarray,
    class_weights: Optional[np.ndarray],
    run_name: str,
) -> Dict:
    """SGD lineal con StandardScaler y mas epocas."""
    print(f"\n{'='*60}")
    print(f"ENTRENANDO SGD NORMALIZADO: {run_name}")
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
            sw = sample_weights_from_class_weights(yb, class_weights)

            if epoch == 0 and step == 0:
                clf.partial_fit(xb, yb, classes=classes, sample_weight=sw)
            else:
                clf.partial_fit(xb, yb, sample_weight=sw)

            now = time.time()
            if now - last_print >= PROGRESS_INTERVAL or (epoch == SGD_EPOCHS - 1 and step == total_steps - 1):
                pct = 100.0 * (epoch * total_steps + step + 1) / (SGD_EPOCHS * total_steps)
                print(f"  [{pct:5.1f}%] Epoch {epoch+1}/{SGD_EPOCHS} step {step+1}/{total_steps}")
                last_print = now

    y_pred = clf.predict(X_eval)
    metrics = evaluar(f"EVAL {run_name}", y_eval, y_pred)
    return metrics


# =====================
# MLP PYTORCH
# =====================


def entrenar_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval: np.ndarray, y_eval: np.ndarray,
    class_weights: Optional[np.ndarray],
    loss_type: str,
    run_name: str,
    focal_gamma: float = 2.0,
) -> Dict:
    """Entrena MLP con PyTorch.

    loss_type: 'ce' (CrossEntropy), 'focal' (FocalLoss)
    """
    print(f"\n{'='*60}")
    print(f"ENTRENANDO MLP: {run_name}")
    print(f"  Train: {len(y_train):,} | Eval: {len(y_eval):,}")
    print(f"  Hidden: {MLP_HIDDEN} | Dropout: {MLP_DROPOUT}")
    print(f"  LR: {MLP_LR} | Epochs: {MLP_EPOCHS} | Loss: {loss_type}")
    if loss_type == "focal":
        print(f"  Focal gamma: {focal_gamma}")
    if class_weights is not None:
        cw_str = ", ".join(f"{AGE_GROUPS[i]}={class_weights[i]:.3f}" for i in range(NUM_CLASSES))
        print(f"  Pesos de clase: {cw_str}")
    print(f"{'='*60}")

    input_dim = X_train.shape[1]

    model = AgeMLP(input_dim, MLP_HIDDEN, NUM_CLASSES, MLP_DROPOUT).to(DEVICE)

    # Loss
    if loss_type == "focal":
        alpha_tensor = None
        if class_weights is not None:
            alpha_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        criterion = FocalLoss(gamma=focal_gamma, alpha=alpha_tensor)
    else:
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)

    # Cosine annealing schedule
    n_train = len(y_train)
    steps_per_epoch = math.ceil(n_train / MLP_BATCH_SIZE)
    total_steps = MLP_EPOCHS * steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    last_print = time.time()
    global_step = 0

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
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == yb).sum().item()
            epoch_total += len(yb)
            global_step += 1

        avg_loss = epoch_loss / epoch_total
        avg_acc = epoch_correct / epoch_total
        lr = scheduler.get_last_lr()[0]

        now = time.time()
        if now - last_print >= PROGRESS_INTERVAL or epoch == 0 or epoch == MLP_EPOCHS - 1:
            print(f"  Epoch {epoch+1}/{MLP_EPOCHS}: loss={avg_loss:.4f} "
                  f"train_acc={avg_acc:.4f} lr={lr:.6f}")
            last_print = now

    # Eval
    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(y_eval), MLP_BATCH_SIZE):
            end = min(start + MLP_BATCH_SIZE, len(y_eval))
            xb = torch.tensor(X_eval[start:end], dtype=torch.float32, device=DEVICE)
            logits = model(xb)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    y_pred = np.concatenate(all_preds)
    metrics = evaluar(f"EVAL {run_name}", y_eval, y_pred)
    return metrics


# =====================
# AGREGACION POR USUARIO
# =====================


def agregar_por_usuario(
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


# =====================
# MAIN
# =====================


def main():
    print("=" * 70)
    print("CLASIFICADOR EDAD MEJORADO - ACTIVACIONES GPT-2")
    print("=" * 70)

    # 1. Cargar activaciones pre-extraidas
    last_token, mean_token, labels, authors, hidden_size = cargar_activaciones()
    has_author = authors is not None

    # Distribucion
    print(f"\nDistribucion de clases ({len(labels):,} comentarios):")
    for i, group in enumerate(AGE_GROUPS):
        count = int((labels == i).sum())
        print(f"  {group}: {count:,} ({100*count/len(labels):.1f}%)")

    # 2. Cargar splits
    train_idx, eval_idx, test_idx = cargar_splits(len(labels))
    y_train_c = labels[train_idx]
    y_eval_c = labels[eval_idx]

    print(f"\nSplit comentarios:")
    for i, group in enumerate(AGE_GROUPS):
        tr_n = int((y_train_c == i).sum())
        ev_n = int((y_eval_c == i).sum())
        print(f"  {group}: train={tr_n:,} eval={ev_n:,}")

    # Pesos de clase
    class_weights = calcular_pesos_clase(y_train_c)
    cw_str = ", ".join(f"{AGE_GROUPS[i]}={class_weights[i]:.3f}" for i in range(NUM_CLASSES))
    print(f"\nPesos de clase (automaticos): {cw_str}")

    all_results = {}

    # ==============================
    # A) NIVEL COMENTARIO
    # ==============================
    feature_map = {"last_token": last_token, "mean": mean_token}

    for pooling_name in COMMENT_POOLINGS:
        raw_feats = feature_map[pooling_name]

        print(f"\n{'#'*70}")
        print(f"# COMENTARIO - pooling: {pooling_name}")
        print(f"{'#'*70}")

        # 3. Normalizar features (fit en train, transform ambos)
        print(f"\nFitting StandardScaler en train ({pooling_name})...")
        scaler = fit_scaler_incremental(raw_feats, train_idx)
        print(f"Transformando train...")
        X_train = transform_in_batches(raw_feats, train_idx, scaler)
        print(f"Transformando eval...")
        X_eval = transform_in_batches(raw_feats, eval_idx, scaler)

        # --- SGD normalizado sin balanceo ---
        key = f"comentario_{pooling_name}_sgd_norm_sin_balanceo"
        metrics = entrenar_sgd_normalizado(
            X_train, y_train_c, X_eval, y_eval_c,
            class_weights=None, run_name=key,
        )
        all_results[key] = metrics

        # --- SGD normalizado con balanceo ---
        key = f"comentario_{pooling_name}_sgd_norm_balanceado"
        metrics = entrenar_sgd_normalizado(
            X_train, y_train_c, X_eval, y_eval_c,
            class_weights=class_weights, run_name=key,
        )
        all_results[key] = metrics

        # --- MLP + CE sin balanceo ---
        key = f"comentario_{pooling_name}_mlp_ce_sin_balanceo"
        metrics = entrenar_mlp(
            X_train, y_train_c, X_eval, y_eval_c,
            class_weights=None, loss_type="ce", run_name=key,
        )
        all_results[key] = metrics

        # --- MLP + CE con balanceo ---
        key = f"comentario_{pooling_name}_mlp_ce_balanceado"
        metrics = entrenar_mlp(
            X_train, y_train_c, X_eval, y_eval_c,
            class_weights=class_weights, loss_type="ce", run_name=key,
        )
        all_results[key] = metrics

        # --- MLP + Focal Loss (con distintos gamma) ---
        for gamma in FOCAL_GAMMAS:
            key = f"comentario_{pooling_name}_mlp_focal_g{gamma:.0f}"
            metrics = entrenar_mlp(
                X_train, y_train_c, X_eval, y_eval_c,
                class_weights=class_weights, loss_type="focal",
                run_name=key, focal_gamma=gamma,
            )
            all_results[key] = metrics

        # Liberar memoria
        del X_train, X_eval

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
            print(f"Agregando features por usuario...")
            X_train_user, y_train_user = agregar_por_usuario(
                authors, raw_feats, labels, set(train_auth),
            )
            X_eval_user, y_eval_user = agregar_por_usuario(
                authors, raw_feats, labels, set(eval_auth),
            )

            # Normalizar
            print(f"Normalizando features de usuario...")
            scaler_u = StandardScaler()
            X_train_user = scaler_u.fit_transform(X_train_user)
            X_eval_user = scaler_u.transform(X_eval_user)

            # Pesos de clase a nivel usuario
            user_class_weights = calcular_pesos_clase(y_train_user)

            # --- SGD normalizado sin balanceo ---
            key = f"usuario_{pooling_name}_sgd_norm_sin_balanceo"
            metrics = entrenar_sgd_normalizado(
                X_train_user, y_train_user, X_eval_user, y_eval_user,
                class_weights=None, run_name=key,
            )
            all_results[key] = metrics

            # --- SGD normalizado con balanceo ---
            key = f"usuario_{pooling_name}_sgd_norm_balanceado"
            metrics = entrenar_sgd_normalizado(
                X_train_user, y_train_user, X_eval_user, y_eval_user,
                class_weights=user_class_weights, run_name=key,
            )
            all_results[key] = metrics

            # --- MLP + CE sin balanceo ---
            key = f"usuario_{pooling_name}_mlp_ce_sin_balanceo"
            metrics = entrenar_mlp(
                X_train_user, y_train_user, X_eval_user, y_eval_user,
                class_weights=None, loss_type="ce", run_name=key,
            )
            all_results[key] = metrics

            # --- MLP + CE con balanceo ---
            key = f"usuario_{pooling_name}_mlp_ce_balanceado"
            metrics = entrenar_mlp(
                X_train_user, y_train_user, X_eval_user, y_eval_user,
                class_weights=user_class_weights, loss_type="ce", run_name=key,
            )
            all_results[key] = metrics

            # --- MLP + Focal Loss ---
            for gamma in FOCAL_GAMMAS:
                key = f"usuario_{pooling_name}_mlp_focal_g{gamma:.0f}"
                metrics = entrenar_mlp(
                    X_train_user, y_train_user, X_eval_user, y_eval_user,
                    class_weights=user_class_weights, loss_type="focal",
                    run_name=key, focal_gamma=gamma,
                )
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
    print("COMPLETADO - Test reservado para uso futuro")
    print("=" * 70)


if __name__ == "__main__":
    main()
