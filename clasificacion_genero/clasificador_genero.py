"""
Clasificador de género usando representaciones de SAE.

Este script:
1. Carga la SAE entrenada
2. Extrae características latentes (códigos sparse) de los textos
3. Entrena un clasificador supervisado de género en modo streaming
4. Evalúa el rendimiento y guarda el modelo
"""
# SGDClassifier de scikit-learn, con loss log_loss (equivale a regresión logística lineal entrenada de forma incremental con partial_fit):
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import dotenv
import joblib
import numpy as np
import torch
from datasets import Dataset, disable_progress_bar
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tiny_sae import Sae
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocesamiento import preparar_dataset_para_sae

dotenv.load_dotenv()
disable_progress_bar()


# =====================
# CONFIGURACIÓN
# =====================

MODEL = "openai-community/gpt2"
CONTEXT_LEN = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Rutas
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"
PATH_SAE = "sae-ckpts/sae-gpt2-genero"  # Donde se guardó la SAE entrenada
TEXT_COLUMN = "body"

# Para testing/desarrollo, limitar comentarios
MAX_COMMENTS = None  # None = todos

# IMPORTANTE: Clasificación BINARIA - Solo usa género 'f' y 'm'
# Se excluye 'unknown' del entrenamiento y evaluación

# Splits
TEST_SIZE = 0.2
EVAL_SIZE = 0.1
RANDOM_STATE = 42

# Entrenamiento incremental
FEATURE_BATCH_SIZE = 16
TRAIN_EPOCHS = 1
SGD_ALPHA = 1e-5
PROGRESS_MIN_INTERVAL = 5
FEMALE_CLASS_WEIGHT = 1.2
MALE_CLASS_WEIGHT = 1.0

# Estrategias de agregación de tokens para representar cada texto.
# - "mean": promedio de activaciones sobre tokens válidos (actual)
# - "last_token": usa solo el último token no-padding
# Se ejecutan en orden para comparar ambos enfoques en una sola corrida.
POOLING_MODES_TO_RUN = ["mean", "last_token"]


# =====================
# CARGA DE DATOS
# =====================


def cargar_dataset_genero() -> Dataset:
    """Carga comentarios + autores con género usando función centralizada."""

    df_comentarios_con_genero, _ = preparar_dataset_para_sae(
        path_comentarios=PATH_COMENTARIOS,
        path_autores=PATH_AUTORES,
        max_comments=MAX_COMMENTS,
        solo_genero_conocido=True,
    )

    if TEXT_COLUMN in df_comentarios_con_genero.columns and TEXT_COLUMN != "text":
        df_comentarios_con_genero = df_comentarios_con_genero.rename(
            columns={TEXT_COLUMN: "text"}
        )

    df_comentarios_con_genero = df_comentarios_con_genero.dropna(subset=["text"]).reset_index(
        drop=True
    )

    # Garantía explícita: solo comentarios con género conocido (m/f), sin NaN en gender_clean.
    df_comentarios_con_genero = df_comentarios_con_genero.dropna(subset=["gender_clean"])
    df_comentarios_con_genero = df_comentarios_con_genero[
        df_comentarios_con_genero["gender_clean"].isin(["m", "f"])
    ].reset_index(drop=True)

    generos_presentes = set(df_comentarios_con_genero["gender_clean"].unique())
    assert generos_presentes <= {"m", "f"}, (
        f"ERROR: gender_clean contiene valores inesperados: {generos_presentes - {'m', 'f'}}"
    )
    dist = df_comentarios_con_genero["gender_clean"].value_counts().to_dict()
    print(f"\n[VERIFICACIÓN] Géneros presentes (solo m/f): {dist}")
    print(f"[VERIFICACIÓN] Nulos en gender_clean: {df_comentarios_con_genero['gender_clean'].isna().sum()}")

    return Dataset.from_pandas(df_comentarios_con_genero[["text", "gender_clean"]])


# =====================
# UTILIDADES
# =====================


def _dividir_indices_estratificados(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve índices train/eval/test manteniendo proporciones de clase."""
    if TEST_SIZE + EVAL_SIZE >= 1.0:
        raise ValueError("TEST_SIZE + EVAL_SIZE debe ser < 1.0")

    indices = np.arange(labels.shape[0])
    train_eval_idx, test_idx, y_train_eval, _ = train_test_split(
        indices,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    eval_relative_size = EVAL_SIZE / (1.0 - TEST_SIZE)
    train_idx, eval_idx = train_test_split(
        train_eval_idx,
        test_size=eval_relative_size,
        random_state=RANDOM_STATE,
        stratify=y_train_eval,
    )

    return train_idx, eval_idx, test_idx


def _calcular_pesos_clase(y_train: np.ndarray) -> Dict[int, float]:
    """Devuelve pesos fijos suaves para evitar diferencias grandes entre clases."""
    n_f = int((y_train == 0).sum())
    n_m = int((y_train == 1).sum())

    if n_f == 0 or n_m == 0:
        raise ValueError("El split de train debe contener ejemplos de ambas clases.")

    return {
        0: float(FEMALE_CLASS_WEIGHT),
        1: float(MALE_CLASS_WEIGHT),
    }


def _pool_features(
    top_acts: torch.Tensor,
    top_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    sae: Sae,
    pooling_mode: str,
) -> torch.Tensor:
    """Agrega activaciones sparse de cada token en una única representación por texto."""
    if pooling_mode not in {"mean", "last_token"}:
        raise ValueError(f"pooling_mode='{pooling_mode}' no válido. Usa 'mean' o 'last_token'.")

    batch_size, seq_len, k = top_acts.shape
    attention_mask_sae = attention_mask.to(sae.device)

    pooled = torch.zeros(
        batch_size,
        sae.cfg.num_latents,
        device=sae.device,
        dtype=top_acts.dtype,
    )

    if pooling_mode == "mean":
        token_mask = attention_mask_sae.unsqueeze(-1).to(top_acts.dtype)
        top_acts_masked = top_acts * token_mask

        batch_ids = torch.arange(batch_size, device=sae.device).view(-1, 1, 1)
        batch_ids = batch_ids.expand(-1, seq_len, k)

        pooled.index_put_(
            (batch_ids.reshape(-1), top_indices.reshape(-1)),
            top_acts_masked.reshape(-1),
            accumulate=True,
        )

        valid_tokens = attention_mask_sae.sum(dim=1).clamp(min=1).to(top_acts.dtype)
        pooled = pooled / valid_tokens.unsqueeze(1)
    else:
        last_pos = attention_mask_sae.sum(dim=1).clamp(min=1).long() - 1
        batch_ids = torch.arange(batch_size, device=sae.device)

        last_top_acts = top_acts[batch_ids, last_pos, :]
        last_top_indices = top_indices[batch_ids, last_pos, :]

        pooled.index_put_(
            (
                batch_ids.view(-1, 1).expand(-1, k).reshape(-1),
                last_top_indices.reshape(-1),
            ),
            last_top_acts.reshape(-1),
            accumulate=True,
        )

    return pooled


def iterar_features_sae(
    dataset: Dataset,
    sae: Sae,
    model: AutoModelForCausalLM,
    tokenizer,
    pooling_mode: str,
    desc: str,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Genera batches de (features, labels) sin acumular todo en memoria."""
    hookpoint = model.get_submodule(sae.cfg.hookpoint)
    captured_activations = None

    def hook(module, inputs, outputs):
        nonlocal captured_activations
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        captured_activations = outputs.detach()

    handle = hookpoint.register_forward_hook(hook)
    total = len(dataset)
    total_steps = math.ceil(total / FEATURE_BATCH_SIZE)

    try:
        with torch.no_grad():
            for start in tqdm(
                range(0, total, FEATURE_BATCH_SIZE),
                total=total_steps,
                desc=desc,
                mininterval=PROGRESS_MIN_INTERVAL,
                dynamic_ncols=True,
            ):
                end = min(start + FEATURE_BATCH_SIZE, total)
                batch = dataset[start:end]

                textos = batch["text"]
                labels = np.fromiter(
                    (0 if g == "f" else 1 for g in batch["gender_clean"]),
                    dtype=np.int64,
                    count=len(textos),
                )

                tokens = tokenizer(
                    textos,
                    max_length=CONTEXT_LEN,
                    truncation=True,
                    padding="max_length",
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(model.device)
                attention_mask = tokens["attention_mask"].to(model.device)

                _ = model(input_ids=input_ids, attention_mask=attention_mask)

                acts = captured_activations.to(sae.dtype).to(sae.device)
                top_acts, top_indices = sae.encode(acts)
                pooled = _pool_features(
                    top_acts=top_acts,
                    top_indices=top_indices,
                    attention_mask=attention_mask,
                    sae=sae,
                    pooling_mode=pooling_mode,
                )

                yield pooled.to(torch.float32).cpu().numpy(), labels
    finally:
        handle.remove()


def _evaluar_split(nombre_split: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Imprime métricas completas para un split."""
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precision_por_clase = precision_score(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    recall_por_clase = recall_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    f1_por_clase = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)

    print(f"\n=== Evaluación en {nombre_split} ===")
    print(f"Accuracy: {acc:.4f}")
    print(
        "Macro avg: "
        f"precision={precision_macro:.4f}, recall={recall_macro:.4f}, f1={f1_macro:.4f}"
    )
    print(
        "Weighted avg: "
        f"precision={precision_weighted:.4f}, recall={recall_weighted:.4f}, f1={f1_weighted:.4f}"
    )

    print("\nMétricas por género:")
    print(
        "  female (0): "
        f"precision={precision_por_clase[0]:.4f}, "
        f"recall={recall_por_clase[0]:.4f}, "
        f"f1={f1_por_clase[0]:.4f}"
    )
    print(
        "  male   (1): "
        f"precision={precision_por_clase[1]:.4f}, "
        f"recall={recall_por_clase[1]:.4f}, "
        f"f1={f1_por_clase[1]:.4f}"
    )

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["female", "male"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# =====================
# ENTRENAMIENTO DEL CLASIFICADOR
# =====================


def entrenar_clasificador_streaming(
    dataset: Dataset,
    sae: Sae,
    model: AutoModelForCausalLM,
    tokenizer,
    pooling_mode: str,
) -> SGDClassifier:
    """Entrena un clasificador de género sin acumular todas las features en RAM."""
    labels_all = np.fromiter(
        (0 if g == "f" else 1 for g in dataset["gender_clean"]),
        dtype=np.int8,
        count=len(dataset),
    )

    train_idx, eval_idx, test_idx = _dividir_indices_estratificados(labels_all)
    train_dataset = dataset.select(train_idx.tolist())
    eval_dataset = dataset.select(eval_idx.tolist())

    print("\nDividiendo en train/eval/test...")
    print(
        f"Train: ({len(train_dataset)}, {sae.cfg.num_latents}), "
        f"Eval: ({len(eval_dataset)}, {sae.cfg.num_latents}), "
        f"Test: ({len(test_idx)}, {sae.cfg.num_latents})"
    )

    class_weights = _calcular_pesos_clase(labels_all[train_idx])
    print(
        "Pesos de clase (train): "
        f"female(0)={class_weights[0]:.4f}, male(1)={class_weights[1]:.4f}"
    )

    clf = SGDClassifier(
        loss="log_loss",
        alpha=SGD_ALPHA,
        max_iter=1,
        tol=None,
        random_state=RANDOM_STATE,
    )

    print("\nEntrenando clasificador incremental (SGD con log_loss)...")
    print(f"Train batches por época: {math.ceil(len(train_dataset) / FEATURE_BATCH_SIZE):,}")

    first_batch = True
    classes = np.array([0, 1], dtype=np.int64)

    for epoch in range(TRAIN_EPOCHS):
        print(f"\nÉpoca {epoch + 1}/{TRAIN_EPOCHS}")
        for x_batch, y_batch in iterar_features_sae(
            dataset=train_dataset,
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            pooling_mode=pooling_mode,
            desc=f"Train ({pooling_mode})",
        ):
            sample_weight = np.where(y_batch == 0, class_weights[0], class_weights[1]).astype(
                np.float32
            )
            if first_batch:
                clf.partial_fit(
                    x_batch,
                    y_batch,
                    classes=classes,
                    sample_weight=sample_weight,
                )
                first_batch = False
            else:
                clf.partial_fit(x_batch, y_batch, sample_weight=sample_weight)

    print("\nEvaluando en eval set...")
    y_true_eval = []
    y_pred_eval = []
    for x_batch, y_batch in iterar_features_sae(
        dataset=eval_dataset,
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        pooling_mode=pooling_mode,
        desc=f"Eval ({pooling_mode})",
    ):
        y_true_eval.append(y_batch)
        y_pred_eval.append(clf.predict(x_batch))

    y_true_eval_arr = np.concatenate(y_true_eval)
    y_pred_eval_arr = np.concatenate(y_pred_eval)
    _evaluar_split("eval", y_true_eval_arr, y_pred_eval_arr)

    print("\nTest set reservado (sin evaluación final en esta corrida).")
    return clf


# =====================
# MAIN
# =====================


def main():
    print("=" * 60)
    print("CLASIFICADOR DE GÉNERO CON SAE")
    print("=" * 60)

    dataset = cargar_dataset_genero()

    if not Path(PATH_SAE).exists():
        raise FileNotFoundError(
            f"No se encuentra la SAE en {PATH_SAE}. "
            "Primero ejecuta sae_genero.py para entrenar la SAE."
        )

    print(f"\nCargando SAE desde {PATH_SAE}...")
    sae = Sae.load_from_disk(PATH_SAE, device=DEVICE)
    print(f"SAE cargada: {sae.cfg.num_latents} latentes, k={sae.cfg.k}")

    print("\nCargando modelo GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": DEVICE},
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    os.makedirs("modelos", exist_ok=True)

    for i, pooling_mode in enumerate(POOLING_MODES_TO_RUN, start=1):
        print("\n" + "-" * 60)
        print(f"[RUN {i}/{len(POOLING_MODES_TO_RUN)}] pooling={pooling_mode}")
        print("-" * 60)

        clf = entrenar_clasificador_streaming(
            dataset=dataset,
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            pooling_mode=pooling_mode,
        )

        mode_output_path = f"modelos/clasificador_genero_{pooling_mode}.pkl"
        joblib.dump(clf, mode_output_path)
        print(f"\nClasificador guardado en: {mode_output_path}")

        if i == 1:
            default_output_path = "modelos/clasificador_genero.pkl"
            joblib.dump(clf, default_output_path)
            print(f"Clasificador por defecto actualizado en: {default_output_path}")

    print("\n" + "=" * 60)
    print("¡Entrenamiento completado!")
    print("=" * 60)
    print("\nPuedes usar este clasificador para inferir género de nuevos textos.")
    print("Las características se extraen con la SAE y luego se clasifican.")


if __name__ == "__main__":
    main()
