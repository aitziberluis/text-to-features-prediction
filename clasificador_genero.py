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
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from tiny_sae import Sae
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocesamiento import preparar_dataset_para_sae

dotenv.load_dotenv()


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

# Test split
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =====================
# CARGA DE DATOS
# =====================


def cargar_dataset_genero() -> Dataset:
    """Carga comentarios + autores con género usando función centralizada."""

    # Usar la función centralizada de preprocesamiento
    df_comentarios_con_genero, _ = preparar_dataset_para_sae(
        path_comentarios=PATH_COMENTARIOS,
        path_autores=PATH_AUTORES,
        max_comments=MAX_COMMENTS,
        solo_genero_conocido=True
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
    dataset: Dataset, sae: Sae, model: AutoModelForCausalLM, tokenizer
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae características latentes de la SAE para cada comentario.

    Returns
    -------
    features: np.ndarray de forma (n_samples, num_latents)
        Activaciones sparse agregadas por comentario.
    labels: np.ndarray de forma (n_samples,)
        Género codificado como 0 (f) o 1 (m).
    """

    print("Tokenizando dataset...")

    def _tokenize_fn(x: Dict[str, list]):
        return tokenizer(
            x["text"],
            max_length=CONTEXT_LEN,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )

    tokenized = dataset.map(
        _tokenize_fn, batched=True, batch_size=32, num_proc=4, load_from_cache_file=True
    )

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

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

    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extrayendo features"):
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)

                # Forward pass por el modelo para capturar activaciones
                _ = model(input_ids, attention_mask=attention_mask)

                # Activaciones capturadas en la capa intermedia
                acts = captured_activations.to(sae.dtype).to(sae.device)

                # Codificar con la SAE
                top_acts, top_indices = sae.encode(acts)

                # Agregar a nivel de secuencia (mean pooling sobre tokens)
                # top_acts shape: (batch, seq_len, k)
                # Convertimos a representación densa para simplicidad
                batch_size, seq_len, k = top_acts.shape

                # Crear representación sparse densa
                dense_acts = torch.zeros(
                    batch_size, seq_len, sae.cfg.num_latents, device=sae.device
                )
                for b in range(batch_size):
                    for s in range(seq_len):
                        dense_acts[b, s, top_indices[b, s]] = top_acts[b, s]

                # Mean pooling sobre tokens (ignorando padding si es necesario)
                # Aquí simplemente promediamos sobre la dimensión de secuencia
                pooled = dense_acts.mean(dim=1)  # (batch, num_latents)

                features_list.append(pooled.cpu().numpy())

                # Extraer labels (gender_clean)
                genders = [dataset[int(idx)]["gender_clean"] for idx in batch["__index_level_0__"]]
                labels = np.array([0 if g == "f" else 1 for g in genders])
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

    print("\nDividiendo en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("\nEntrenando clasificador de género (Logistic Regression)...")
    clf = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced", verbose=1
    )
    clf.fit(X_train, y_train)

    print("\nEvaluando en test set...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["female", "male"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

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

    # 4. Extraer características
    features, labels = extraer_caracteristicas_sae(dataset, sae, model, tokenizer)

    # 5. Entrenar clasificador
    clf = entrenar_clasificador(features, labels)

    # 6. Guardar clasificador
    import joblib

    output_path = "modelos/clasificador_genero.pkl"
    os.makedirs("modelos", exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"\nClasificador guardado en: {output_path}")

    print("\n" + "=" * 60)
    print("¡Entrenamiento completado!")
    print("=" * 60)
    print("\nPuedes usar este clasificador para inferir género de nuevos textos.")
    print("Las características se extraen con la SAE y luego se clasifican.")


if __name__ == "__main__":
    main()
