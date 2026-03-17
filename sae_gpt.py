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

MODEL = "openai-community/gpt2"  # puedes cambiar a otro modelo causal
CONTEXT_LEN = 512  # un poco menor que 1024 para ahorrar memoria
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Rutas a tus datos
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"

# Columna de texto en el CSV de comentarios
TEXT_COLUMN = "body"  # en tu CSV hemos visto que es 'body'


# Limitar nº de comentarios para pruebas (None = todos)
MAX_COMMENTS = None  # p.ej. 200_000 para un subset

# IMPORTANTE: Este experimento usa SOLO género 'f' y 'm'
# Se excluye 'unknown' porque es una clasificación binaria


# =====================
# CARGA Y PREPARACIÓN DE DATOS
# =====================


def cargar_dataset_genero() -> Dataset:
	"""Carga TODOS los comentarios y prepara un Dataset solo con el texto.

	La SAE es un modelo no supervisado, así que no necesita
	información de género; se entrena sobre todo el corpus para
	que luego pueda reutilizarse en tareas de género, edad, etc.
	"""

	# Cargar comentarios brutos (limpieza general de author/body)
	df_comentarios = cargar_comentarios(PATH_COMENTARIOS, nrows=MAX_COMMENTS)

	# Renombrar la columna de texto a 'text' si es necesario
	if TEXT_COLUMN in df_comentarios.columns and TEXT_COLUMN != "text":
		df_comentarios = df_comentarios.rename(columns={TEXT_COLUMN: "text"})

	# Limpiar NaNs en la columna de texto
	df_comentarios = df_comentarios.dropna(subset=["text"]).reset_index(drop=True)

	# Crear Dataset de HuggingFace solo con la columna de texto
	dataset = Dataset.from_pandas(df_comentarios[["text"]])

	return dataset


# =====================
# TOKENIZACIÓN Y MODELO
# =====================


def preparar_modelo_y_datos(dataset: Dataset):
	"""Prepara el tokenizer, el modelo y tokeniza el dataset."""

	print("Cargando tokenizer y modelo...")
	tokenizer = AutoTokenizer.from_pretrained(MODEL)
	gpt = AutoModelForCausalLM.from_pretrained(
		MODEL,
		device_map={"": DEVICE},
		torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
	)

	# Aseguramos que el tokenizer tiene token de padding
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

	# Podemos opcionalmente filtrar secuencias que no llenen el contexto completo,
	# pero como usamos padding a longitud fija, no es necesario.

	return tokenizer, gpt, tokenized


# =====================
# CONFIGURACIÓN Y ENTRENAMIENTO DE LA SAE
# =====================


def entrenar_sae(dataset: Dataset):
	"""Entrena una SAE sobre las activaciones de GPT2 usando tus textos."""

	_, gpt, tokenized = preparar_modelo_y_datos(dataset)

	print("Configurando SAE...")
	sae_cfg = SaeConfig(
		# Para GPT-2 small el tamaño de representación interna es 768
		d_in=768,
		num_latents=2**14,
		k=64,
		hookpoint="transformer.h.8",  # misma capa que en tu ejemplo
	)

	sae = Sae(sae_cfg, device=DEVICE)

	print("Configurando entrenamiento de la SAE...")
	train_cfg = TrainConfig(
		wandb_project="tiny-sae-genero",
		wandb_name="sae-gpt2-genero",
		save_every_n_tokens=10_000_000,
		optimize_every_n_tokens=8192,
		model_batch_size=16,
		mask_first_n_tokens=1,
	)

	print("Iniciando entrenamiento de la SAE...")
	train_sae(
		sae=sae,
		model=gpt,
		token_iterator=tokenized,
		train_cfg=train_cfg,
		use_wandb=True,
	)

	# Guardamos la SAE en disco para usarla luego como extractor de características
	output_dir = "sae-ckpts/sae-gpt2-genero"
	os.makedirs(output_dir, exist_ok=True)
	print(f"Guardando SAE entrenada en {output_dir} ...")
	sae.save_to_disk(output_dir)

	print("Entrenamiento de SAE finalizado.")


if __name__ == "__main__":
	print("Cargando dataset de comentarios con género...")
	ds = cargar_dataset_genero()

	print("Tamaño del dataset (comentarios con género m/f):", len(ds))

	# Entrena la SAE sobre tus textos
	entrenar_sae(ds)

	print("\nA continuación, el siguiente paso será usar esta SAE para extraer\n"
	      "características de los textos (códigos latentes) y entrenar un\n"
	      "clasificador supervisado de género (por ejemplo, una regresión\n"
	      "logística o una pequeña red neuronal) usando esas características.")
