"""
Clasificador de género usando representaciones de SAE.

Este script:
1. Carga la SAE entrenada
2. Extrae características latentes (códigos sparse) de los textos
3. Entrena un clasificador supervisado de género
4. Evalúa el rendimiento y guarda el modelo
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import dotenv
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, disable_progress_bar
from sklearn.linear_model import LogisticRegression
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
from torch.utils.data import DataLoader
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

    # Usar la función centralizada de preprocesamiento
    # solo_genero_conocido=True → SOLO 'f' y 'm', excluye 'unknown'
    df_comentarios_con_genero, _ = preparar_dataset_para_sae(
        path_comentarios=PATH_COMENTARIOS,
        path_autores=PATH_AUTORES,
        max_comments=MAX_COMMENTS,
        solo_genero_conocido=True  # ← CLASIFICACIÓN BINARIA: solo 'f' y 'm'
    )
    
    # Renombrar la columna 'body' a 'text' si es necesario
    if TEXT_COLUMN in df_comentarios_con_genero.columns and TEXT_COLUMN != "text":
        df_comentarios_con_genero = df_comentarios_con_genero.rename(columns={TEXT_COLUMN: "text"})
    
    # Limpiar NaNs en la columna de texto
    df_comentarios_con_genero = df_comentarios_con_genero.dropna(subset=["text"]).reset_index(drop=True)
    
    print(f"\nDistribución final: {df_comentarios_con_genero['gender_clean'].value_counts().to_dict()}")

    dataset = Dataset.from_pandas(df_comentarios_con_genero[["text", "gender_clean"]])
    return dataset


# =====================
# EXTRACCIÓN DE CARACTERÍSTICAS CON SAE
# =====================


def extraer_caracteristicas_sae(
    dataset: Dataset,
    sae: Sae,
    model: AutoModelForCausalLM,
    tokenizer,
    pooling_mode: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae características latentes de la SAE para cada comentario.

    Returns
    -------
    features: np.ndarray de forma (n_samples, num_latents)
        Activaciones sparse agregadas por comentario.
    labels: np.ndarray de forma (n_samples,)
        Género codificado como 0 (f) o 1 (m).
    """

    print(f"Tokenizando dataset... (pooling={pooling_mode})")

    def _tokenize_fn(x: Dict[str, list]):
        out = tokenizer(
            x["text"],
            max_length=CONTEXT_LEN,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )
        # Guardamos la etiqueta ya codificada para no depender de índices internos.
        out["label"] = [0 if g == "f" else 1 for g in x["gender_clean"]]
        return out

    tokenized = dataset.map(
        _tokenize_fn, batched=True, batch_size=512, num_proc=16, load_from_cache_file=True
    )

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    hookpoint = model.get_submodule(sae.cfg.hookpoint)
    captured_activations = None

    def hook(module, inputs, outputs):
        nonlocal captured_activations
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        captured_activations = outputs.detach()

    handle = hookpoint.register_forward_hook(hook)

    features_list = []
    labels_list = []

    print("Extrayendo características con la SAE...")
    dataloader = DataLoader(tokenized, batch_size=16, shuffle=False)

    if pooling_mode not in {"mean", "last_token"}:
        raise ValueError(
            f"pooling_mode='{pooling_mode}' no válido. Usa 'mean' o 'last_token'."
        )

    try:
        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc="Extrayendo features",
                mininterval=3600,
                maxinterval=3600,
            ):
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)

                # Forward pass por el modelo para capturar activaciones
                _ = model(input_ids, attention_mask=attention_mask)

                # Activaciones capturadas en la capa intermedia
                acts = captured_activations.to(sae.dtype).to(sae.device)

                # Codificar con la SAE
                top_acts, top_indices = sae.encode(acts)

                # Agregación eficiente sin construir tensor denso 3D completo.
                batch_size, seq_len, k = top_acts.shape

                attention_mask_sae = attention_mask.to(sae.device)

                pooled = torch.zeros(
                    batch_size, sae.cfg.num_latents, device=sae.device, dtype=top_acts.dtype
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

                    valid_tokens = (
                        attention_mask_sae.sum(dim=1)
                        .clamp(min=1)
                        .to(top_acts.dtype)
                    )
                    pooled = pooled / valid_tokens.unsqueeze(1)
                else:
                    # Último token real por secuencia (ignorando padding a la derecha).
                    last_pos = attention_mask_sae.sum(dim=1).clamp(min=1).long() - 1
                    batch_ids = torch.arange(batch_size, device=sae.device)

                    last_top_acts = top_acts[batch_ids, last_pos, :]  # (batch, k)
                    last_top_indices = top_indices[batch_ids, last_pos, :]  # (batch, k)

                    pooled.index_put_(
                        (
                            batch_ids.view(-1, 1).expand(-1, k).reshape(-1),
                            last_top_indices.reshape(-1),
                        ),
                        last_top_acts.reshape(-1),
                        accumulate=True,
                    )

                features_list.append(pooled.cpu().numpy())

                labels = batch["label"].cpu().numpy()
                labels_list.append(labels)

    finally:
        handle.remove()

    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)

    print(f"Features extraídas: {features.shape}")
    print(f"Labels: {labels.shape}, distribución: {np.bincount(labels)}")

    return features, labels


# =====================
# ENTRENAMIENTO DEL CLASIFICADOR
# =====================


def entrenar_clasificador(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Entrena un clasificador de género usando las características de la SAE."""

    if TEST_SIZE + EVAL_SIZE >= 1.0:
        raise ValueError("TEST_SIZE + EVAL_SIZE debe ser < 1.0")

    print("\nDividiendo en train/eval/test...")
    # 1) Separamos test del total.
    X_train_eval, _X_test, y_train_eval, _y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # 2) Separamos eval del bloque restante usando proporción relativa.
    eval_relative_size = EVAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train_eval,
        y_train_eval,
        test_size=eval_relative_size,
        random_state=RANDOM_STATE,
        stratify=y_train_eval,
    )

    print(f"Train: {X_train.shape}, Eval: {X_eval.shape}, Test: {_X_test.shape}")

    print("\nEntrenando clasificador de género (Logistic Regression)...")
    clf = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced", verbose=1
    )
    clf.fit(X_train, y_train)

    def _evaluar_split(nombre_split: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        acc = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        precision_weighted = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        precision_por_clase = precision_score(
            y_true, y_pred, average=None, labels=[0, 1], zero_division=0
        )
        recall_por_clase = recall_score(
            y_true, y_pred, average=None, labels=[0, 1], zero_division=0
        )
        f1_por_clase = f1_score(
            y_true, y_pred, average=None, labels=[0, 1], zero_division=0
        )

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

    print("\nEvaluando en eval set...")
    y_pred_eval = clf.predict(X_eval)
    _evaluar_split("eval", y_eval, y_pred_eval)

    print("\nTest set reservado (sin evaluación final en esta corrida).")

    return clf


# =====================
# MAIN
# =====================


def main():
    print("=" * 60)
    print("CLASIFICADOR DE GÉNERO CON SAE")
    print("=" * 60)

    # 1. Cargar dataset
    dataset = cargar_dataset_genero()

    # 2. Cargar SAE entrenada
    if not Path(PATH_SAE).exists():
        raise FileNotFoundError(
            f"No se encuentra la SAE en {PATH_SAE}. "
            "Primero ejecuta sae_genero.py para entrenar la SAE."
        )

    print(f"\nCargando SAE desde {PATH_SAE}...")
    sae = Sae.load_from_disk(PATH_SAE, device=DEVICE)
    print(f"SAE cargada: {sae.cfg.num_latents} latentes, k={sae.cfg.k}")

    # 3. Cargar modelo GPT-2
    print("\nCargando modelo GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": DEVICE},
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    # 4. Entrenar y guardar clasificadores para cada estrategia de pooling
    import joblib
    os.makedirs("modelos", exist_ok=True)

    for i, pooling_mode in enumerate(POOLING_MODES_TO_RUN, start=1):
        print("\n" + "-" * 60)
        print(f"[RUN {i}/{len(POOLING_MODES_TO_RUN)}] pooling={pooling_mode}")
        print("-" * 60)

        features, labels = extraer_caracteristicas_sae(
            dataset=dataset,
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            pooling_mode=pooling_mode,
        )
        clf = entrenar_clasificador(features, labels)

        mode_output_path = f"modelos/clasificador_genero_{pooling_mode}.pkl"
        joblib.dump(clf, mode_output_path)
        print(f"\nClasificador guardado en: {mode_output_path}")

        # Mantener compatibilidad con inferencia existente.
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
