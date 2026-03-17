import math
import os
from typing import Dict

import dotenv
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tiny_sae import Sae, SaeConfig, TrainConfig, train_sae
from tqdm import tqdm

from preprocesamiento import cargar_comentarios


dotenv.load_dotenv()


# =====================
# CONFIGURACIÓN GENERAL
# =====================

MODEL = "Qwen/Qwen3.5-2B"  # modelo base Qwen
CONTEXT_LEN = 512  # longitud de contexto usada para SAE
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Rutas a tus datos
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"

# Columna de texto en el CSV de comentarios
TEXT_COLUMN = "body"

# Limitar nº de comentarios para pruebas (None = todos)
MAX_COMMENTS = None  # p.ej. 200_000 para un subset


# =====================
# CARGA Y PREPARACIÓN DE DATOS
# =====================


def cargar_dataset_texto() -> Dataset:
    """Carga TODOS los comentarios y prepara un Dataset solo con el texto.

    La SAE es un modelo no supervisado, así que se entrena sobre todo el
    corpus de comentarios, sin usar etiquetas de género/edad.
    """

    df_comentarios = cargar_comentarios(PATH_COMENTARIOS, nrows=MAX_COMMENTS)

    if TEXT_COLUMN in df_comentarios.columns and TEXT_COLUMN != "text":
        df_comentarios = df_comentarios.rename(columns={TEXT_COLUMN: "text"})

    df_comentarios = df_comentarios.dropna(subset=["text"]).reset_index(drop=True)

    dataset = Dataset.from_pandas(df_comentarios[["text"]])
    return dataset


# =====================
# TOKENIZACIÓN Y MODELO
# =====================


def preparar_modelo_y_datos(dataset: Dataset):
    """Prepara tokenizer, modelo Qwen y tokeniza el dataset."""

    print("Cargando tokenizer y modelo Qwen...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": DEVICE},
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tokenize_fn(x: Dict[str, list]):
        out = tokenizer(
            x["text"],
            max_length=CONTEXT_LEN,
            truncation=True,
            padding="max_length",
            return_attention_mask=False,
        )
        return out

    print("Tokenizando dataset...")
    tokenized = dataset.map(
        _tokenize_fn,
        batched=True,
        batch_size=32,
        num_proc=4,
        load_from_cache_file=True,
    )

    return tokenizer, model, tokenized


# =====================
# CONFIGURACIÓN Y ENTRENAMIENTO DE LA SAE
# =====================


def entrenar_sae(dataset: Dataset):
    """Entrena una SAE sobre las activaciones de Qwen usando tus textos."""

    _, model, tokenized = preparar_modelo_y_datos(dataset)

    # Inferimos la dimensión de las activaciones a partir de la config del modelo
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        # Fallback genérico: algunos modelos usan "n_embd" u otros campos
        hidden_size = getattr(model.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("No se pudo determinar la dimensión oculta (hidden_size) del modelo Qwen.")

    print(f"Dimensión oculta detectada para Qwen: d_in={hidden_size}")

    print("Configurando SAE...")
    sae_cfg = SaeConfig(
        d_in=hidden_size,
        num_latents=2**14,
        k=64,
        # Para Qwen, un hookpoint típico es una capa intermedia del bloque transformer.
        # Ajusta este valor si quieres enganchar otra capa.
        hookpoint="model.layers.15",
    )

    sae = Sae(sae_cfg, device=DEVICE)

    print("Configurando entrenamiento de la SAE...")
    train_cfg = TrainConfig(
        wandb_project="tiny-sae-qwen",
        wandb_name="sae-qwen3.5-2b",
        save_every_n_tokens=10_000_000,
        optimize_every_n_tokens=8192,
        model_batch_size=16,
        mask_first_n_tokens=1,
    )

    print("Iniciando entrenamiento de la SAE (Qwen)...")
    train_sae(
        sae=sae,
        model=model,
        token_iterator=tokenized,
        train_cfg=train_cfg,
        use_wandb=True,
    )

    output_dir = "sae-ckpts/sae-qwen3.5-2b"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Guardando SAE entrenada en {output_dir} ...")
    sae.save_to_disk(output_dir)

    print("Entrenamiento de SAE con Qwen finalizado.")


if __name__ == "__main__":
    print("Cargando dataset de comentarios...")
    ds = cargar_dataset_texto()

    print("Tamaño del dataset (comentarios):", len(ds))

    entrenar_sae(ds)

    print("\nAhora puedes usar sae-qwen3.5-2b como extractor de características\n"
          "para tareas de género, edad u otras, igual que la SAE con GPT-2.")
