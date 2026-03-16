"""
Clasificador de edad a nivel usuario usando representaciones SAE.

Idea principal:
1. Split estratificado por autor.
2. Cada autor se representa agregando todos sus comentarios en el espacio SAE.
3. Se entrenan dos variantes: sin balanceo y balanceado.
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import dotenv
import joblib
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
from tiny_sae import Sae
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocesamiento import DEFAULT_AGE_GROUP_LABELS, preparar_dataset_para_edad

dotenv.load_dotenv()


MODEL = "openai-community/gpt2"
CONTEXT_LEN = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"
PATH_SAE = "sae-ckpts/sae-gpt2-genero"
TEXT_COLUMN = "body"

MAX_COMMENTS = None

TEST_SIZE = 0.2
EVAL_SIZE = 0.1
RANDOM_STATE = 42

FEATURE_BATCH_SIZE = 16
USER_BATCH_SIZE = 256
TRAIN_EPOCHS = 1
SGD_ALPHA = 1e-5
PROGRESS_MIN_INTERVAL = 5

POOLING_MODES_TO_RUN = ["mean"]
AGE_GROUPS = DEFAULT_AGE_GROUP_LABELS
AGE_GROUP_TO_ID = {label: idx for idx, label in enumerate(AGE_GROUPS)}
ID_TO_AGE_GROUP = {idx: label for label, idx in AGE_GROUP_TO_ID.items()}

RUN_CONFIGS = [
	{
		"name": "sin_balanceo",
		"use_class_weights": False,
	},
	{
		"name": "balanceado",
		"use_class_weights": True,
	},
]


def cargar_dataset_edad_df() -> pd.DataFrame:
	"""Carga comentarios + rango de edad y asegura columnas necesarias."""

	df_comentarios_con_edad, _ = preparar_dataset_para_edad(
		path_comentarios=PATH_COMENTARIOS,
		path_autores=PATH_AUTORES,
		max_comments=MAX_COMMENTS,
		solo_edad_conocida=True,
	)

	if TEXT_COLUMN in df_comentarios_con_edad.columns and TEXT_COLUMN != "text":
		df_comentarios_con_edad = df_comentarios_con_edad.rename(columns={TEXT_COLUMN: "text"})

	required_cols = {"author", "text", "age_group"}
	missing = required_cols.difference(df_comentarios_con_edad.columns)
	if missing:
		raise ValueError(f"Faltan columnas requeridas en dataset: {sorted(missing)}")

	df = (
		df_comentarios_con_edad[["author", "text", "age_group"]]
		.dropna(subset=["author", "text", "age_group"])
		.copy()
	)
	df = df[df["age_group"].isin(AGE_GROUPS)].reset_index(drop=True)

	grupos_presentes = set(df["age_group"].unique())
	assert grupos_presentes <= set(AGE_GROUPS), (
		f"ERROR: age_group contiene valores inesperados: {grupos_presentes - set(AGE_GROUPS)}"
	)
	print(f"[VERIFICACION] Rangos de edad presentes: {df['age_group'].value_counts().to_dict()}")
	print(f"[VERIFICACION] Nulos en age_group: {df['age_group'].isna().sum()}")

	df["author"] = df["author"].astype(str).str.strip()
	df["text"] = df["text"].astype(str)

	n_labels_por_author = df.groupby("author")["age_group"].nunique()
	if int(n_labels_por_author.max()) > 1:
		conflictivos = int((n_labels_por_author > 1).sum())
		raise ValueError(f"Hay {conflictivos} autores con mas de una etiqueta de edad.")

	user_df = df[["author", "age_group"]].drop_duplicates("author")
	user_dist = user_df["age_group"].value_counts().to_dict()

	print("\nResumen dataset por comentario:")
	print(f"  - comentarios: {len(df):,}")
	print("\nResumen dataset por usuario:")
	print(f"  - usuarios: {len(user_df):,}")
	print(f"  - distribucion usuarios: {user_dist}")

	return df


def _dividir_autores_estratificados(
	user_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if TEST_SIZE + EVAL_SIZE >= 1.0:
		raise ValueError("TEST_SIZE + EVAL_SIZE debe ser < 1.0")

	authors = user_df["author"].to_numpy()
	labels = np.fromiter(
		(AGE_GROUP_TO_ID[g] for g in user_df["age_group"].to_numpy()),
		dtype=np.int64,
		count=len(user_df),
	)

	train_eval_authors, test_authors, y_train_eval, _ = train_test_split(
		authors,
		labels,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
		stratify=labels,
	)

	eval_relative_size = EVAL_SIZE / (1.0 - TEST_SIZE)
	train_authors, eval_authors, y_train, _ = train_test_split(
		train_eval_authors,
		y_train_eval,
		test_size=eval_relative_size,
		random_state=RANDOM_STATE,
		stratify=y_train_eval,
	)

	return train_authors, eval_authors, test_authors, y_train


def _calcular_pesos_clase(y_train: np.ndarray) -> Dict[int, float]:
	classes, counts = np.unique(y_train, return_counts=True)
	n_total = y_train.shape[0]
	n_classes = len(classes)
	weights = {}
	for class_id, count in zip(classes, counts):
		weights[int(class_id)] = n_total / (n_classes * float(count))
	return weights


def _pool_features(
	top_acts: torch.Tensor,
	top_indices: torch.Tensor,
	attention_mask: torch.Tensor,
	sae: Sae,
	pooling_mode: str,
) -> torch.Tensor:
	if pooling_mode not in {"mean", "last_token"}:
		raise ValueError(f"pooling_mode='{pooling_mode}' no valido. Usa 'mean' o 'last_token'.")

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


def _iterar_features_por_usuario(
	df_split: pd.DataFrame,
	sae: Sae,
	model: AutoModelForCausalLM,
	tokenizer,
	pooling_mode: str,
	desc: str,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
	if len(df_split) == 0:
		return

	df_sorted = df_split.sort_values("author", kind="mergesort").reset_index(drop=True)

	hookpoint = model.get_submodule(sae.cfg.hookpoint)
	captured_activations = None

	def hook(module, inputs, outputs):
		del module, inputs
		nonlocal captured_activations
		if isinstance(outputs, tuple):
			outputs = outputs[0]
		captured_activations = outputs.detach()

	handle = hookpoint.register_forward_hook(hook)

	current_author = None
	current_label = None
	current_sum = None
	current_count = 0

	user_x_buffer: List[np.ndarray] = []
	user_y_buffer: List[int] = []

	total_comments = len(df_sorted)
	total_steps = math.ceil(total_comments / FEATURE_BATCH_SIZE)

	try:
		with torch.no_grad():
			for start in tqdm(
				range(0, total_comments, FEATURE_BATCH_SIZE),
				total=total_steps,
				desc=desc,
				mininterval=PROGRESS_MIN_INTERVAL,
				dynamic_ncols=True,
			):
				end = min(start + FEATURE_BATCH_SIZE, total_comments)
				batch = df_sorted.iloc[start:end]

				textos = batch["text"].tolist()
				authors = batch["author"].tolist()
				labels = batch["age_group"].tolist()

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
				comment_features = _pool_features(
					top_acts=top_acts,
					top_indices=top_indices,
					attention_mask=attention_mask,
					sae=sae,
					pooling_mode=pooling_mode,
				)
				comment_features = comment_features.to(torch.float32).cpu().numpy()

				for feat, author, label in zip(comment_features, authors, labels):
					y_val = AGE_GROUP_TO_ID[label]

					if current_author is None:
						current_author = author
						current_label = y_val
						current_sum = np.zeros_like(feat, dtype=np.float64)
						current_count = 0

					if author != current_author:
						user_vec = (current_sum / max(current_count, 1)).astype(np.float32, copy=False)
						user_x_buffer.append(user_vec)
						user_y_buffer.append(int(current_label))

						if len(user_x_buffer) >= USER_BATCH_SIZE:
							yield np.vstack(user_x_buffer), np.asarray(user_y_buffer, dtype=np.int64)
							user_x_buffer = []
							user_y_buffer = []

						current_author = author
						current_label = y_val
						current_sum = np.zeros_like(feat, dtype=np.float64)
						current_count = 0

					if y_val != current_label:
						raise ValueError(f"Author '{author}' tiene etiquetas inconsistentes de edad.")

					current_sum += feat
					current_count += 1

		if current_author is not None:
			user_vec = (current_sum / max(current_count, 1)).astype(np.float32, copy=False)
			user_x_buffer.append(user_vec)
			user_y_buffer.append(int(current_label))

		if user_x_buffer:
			yield np.vstack(user_x_buffer), np.asarray(user_y_buffer, dtype=np.int64)
	finally:
		handle.remove()


def _evaluar_split(nombre_split: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	classes = np.arange(len(AGE_GROUPS), dtype=np.int64)
	acc = accuracy_score(y_true, y_pred)
	bal_acc = balanced_accuracy_score(y_true, y_pred)

	precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
	recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
	f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

	precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
	recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
	f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

	print(f"\n=== Evaluacion en {nombre_split} ===")
	print(f"Accuracy: {acc:.4f}")
	print(f"Balanced Accuracy: {bal_acc:.4f}")
	print(
		"Macro avg: "
		f"precision={precision_macro:.4f}, recall={recall_macro:.4f}, f1={f1_macro:.4f}"
	)
	print(
		"Weighted avg: "
		f"precision={precision_weighted:.4f}, recall={recall_weighted:.4f}, f1={f1_weighted:.4f}"
	)

	print("\nClassification Report:")
	print(classification_report(y_true, y_pred, labels=classes, target_names=AGE_GROUPS, zero_division=0))
	print("\nConfusion Matrix:")
	print(confusion_matrix(y_true, y_pred, labels=classes))

	return {
		"accuracy": float(acc),
		"balanced_accuracy": float(bal_acc),
		"precision_macro": float(precision_macro),
		"recall_macro": float(recall_macro),
		"f1_macro": float(f1_macro),
		"precision_weighted": float(precision_weighted),
		"recall_weighted": float(recall_weighted),
		"f1_weighted": float(f1_weighted),
	}


def entrenar_clasificador_por_usuario(
	df_comments: pd.DataFrame,
	sae: Sae,
	model: AutoModelForCausalLM,
	tokenizer,
	pooling_mode: str,
	run_name: str,
	use_class_weights: bool,
) -> Tuple[SGDClassifier, Dict[str, object]]:
	user_df = df_comments[["author", "age_group"]].drop_duplicates("author")

	train_authors, eval_authors, test_authors, y_train_users = _dividir_autores_estratificados(
		user_df=user_df
	)

	split_map = {author: "train" for author in train_authors}
	split_map.update({author: "eval" for author in eval_authors})
	split_map.update({author: "test" for author in test_authors})

	split_labels = df_comments["author"].map(split_map)
	if split_labels.isna().any():
		missing = int(split_labels.isna().sum())
		raise RuntimeError(f"Hay {missing} comentarios sin split asignado.")

	train_df = df_comments.loc[split_labels == "train", ["author", "text", "age_group"]].copy()
	eval_df = df_comments.loc[split_labels == "eval", ["author", "text", "age_group"]].copy()
	test_df = df_comments.loc[split_labels == "test", ["author", "text", "age_group"]].copy()

	print("\nSplit por usuarios:")
	print(f"  - Train: {len(train_authors):,} usuarios, {len(train_df):,} comentarios")
	print(f"  - Eval:  {len(eval_authors):,} usuarios, {len(eval_df):,} comentarios")
	print(f"  - Test:  {len(test_authors):,} usuarios, {len(test_df):,} comentarios")

	class_weights = _calcular_pesos_clase(y_train_users) if use_class_weights else {
		class_id: 1.0 for class_id in range(len(AGE_GROUPS))
	}

	print("\nConfiguracion de pesos de clase:")
	print(f"  - modo: {run_name}")
	print(f"  - use_class_weights: {use_class_weights}")
	print(f"  - pesos train: {class_weights}")

	clf = SGDClassifier(
		loss="log_loss",
		alpha=SGD_ALPHA,
		max_iter=1,
		tol=None,
		random_state=RANDOM_STATE,
		average=True,
	)

	first_batch = True
	classes = np.arange(len(AGE_GROUPS), dtype=np.int64)

	for epoch in range(TRAIN_EPOCHS):
		print(f"\nEpoch {epoch + 1}/{TRAIN_EPOCHS}")
		for x_batch, y_batch in _iterar_features_por_usuario(
			df_split=train_df,
			sae=sae,
			model=model,
			tokenizer=tokenizer,
			pooling_mode=pooling_mode,
			desc=f"Train users edad ({run_name}, {pooling_mode})",
		):
			sample_weight = np.array([class_weights[int(y)] for y in y_batch], dtype=np.float32)
			if first_batch:
				clf.partial_fit(x_batch, y_batch, classes=classes, sample_weight=sample_weight)
				first_batch = False
			else:
				clf.partial_fit(x_batch, y_batch, sample_weight=sample_weight)

	print("\nEvaluando en eval (agregado por usuario)...")
	y_eval_true_parts = []
	y_eval_pred_parts = []
	for x_batch, y_batch in _iterar_features_por_usuario(
		df_split=eval_df,
		sae=sae,
		model=model,
		tokenizer=tokenizer,
		pooling_mode=pooling_mode,
		desc=f"Eval users edad ({run_name}, {pooling_mode})",
	):
		y_eval_true_parts.append(y_batch)
		y_eval_pred_parts.append(clf.predict(x_batch))

	if not y_eval_true_parts:
		raise RuntimeError("Split de eval vacio al evaluar por usuario.")

	y_eval_true = np.concatenate(y_eval_true_parts)
	y_eval_pred = np.concatenate(y_eval_pred_parts)
	eval_metrics = _evaluar_split("eval usuarios", y_eval_true, y_eval_pred)

	print("\nEvaluando en test (agregado por usuario)...")
	y_test_true_parts = []
	y_test_pred_parts = []
	for x_batch, y_batch in _iterar_features_por_usuario(
		df_split=test_df,
		sae=sae,
		model=model,
		tokenizer=tokenizer,
		pooling_mode=pooling_mode,
		desc=f"Test users edad ({run_name}, {pooling_mode})",
	):
		y_test_true_parts.append(y_batch)
		y_test_pred_parts.append(clf.predict(x_batch))

	if not y_test_true_parts:
		raise RuntimeError("Split de test vacio al evaluar por usuario.")

	y_test_true = np.concatenate(y_test_true_parts)
	y_test_pred = np.concatenate(y_test_pred_parts)
	test_metrics = _evaluar_split("test usuarios", y_test_true, y_test_pred)

	metadata = {
		"run_name": run_name,
		"pooling_mode": pooling_mode,
		"use_class_weights": bool(use_class_weights),
		"age_groups": AGE_GROUPS,
		"eval_metrics": eval_metrics,
		"test_metrics": test_metrics,
		"train_users": int(len(train_authors)),
		"eval_users": int(len(eval_authors)),
		"test_users": int(len(test_authors)),
		"train_comments": int(len(train_df)),
		"eval_comments": int(len(eval_df)),
		"test_comments": int(len(test_df)),
		"feature_batch_size": int(FEATURE_BATCH_SIZE),
		"user_batch_size": int(USER_BATCH_SIZE),
		"train_epochs": int(TRAIN_EPOCHS),
		"alpha": float(SGD_ALPHA),
		"random_state": int(RANDOM_STATE),
	}

	return clf, metadata


def main() -> None:
	print("=" * 70)
	print("CLASIFICADOR DE EDAD POR USUARIO CON SAE")
	print("=" * 70)
	print(f"Rangos de edad usados: {AGE_GROUPS}")

	df_comments = cargar_dataset_edad_df()

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

	total_runs = len(POOLING_MODES_TO_RUN) * len(RUN_CONFIGS)
	run_idx = 0
	for pooling_mode in POOLING_MODES_TO_RUN:
		for run_cfg in RUN_CONFIGS:
			run_idx += 1
			run_name = run_cfg["name"]

			print("\n" + "-" * 70)
			print(f"[RUN {run_idx}/{total_runs}] pooling={pooling_mode} | mode={run_name}")
			print("-" * 70)

			clf, metadata = entrenar_clasificador_por_usuario(
				df_comments=df_comments,
				sae=sae,
				model=model,
				tokenizer=tokenizer,
				pooling_mode=pooling_mode,
				run_name=run_name,
				use_class_weights=bool(run_cfg["use_class_weights"]),
			)

			model_path = f"modelos/clasificador_edad_usuario_{run_name}_{pooling_mode}.pkl"
			joblib.dump(clf, model_path)
			print(f"\nModelo guardado en: {model_path}")

			meta_path = f"modelos/clasificador_edad_usuario_{run_name}_{pooling_mode}_meta.json"
			with open(meta_path, "w", encoding="utf-8") as f:
				json.dump(metadata, f, ensure_ascii=False, indent=2)
			print(f"Metadatos guardados en: {meta_path}")

			if run_idx == 1:
				default_path = "modelos/clasificador_edad_usuario.pkl"
				joblib.dump(clf, default_path)
				print(f"Modelo por defecto actualizado en: {default_path}")

	print("\n" + "=" * 70)
	print("Entrenamiento por usuario de edad completado")
	print("=" * 70)


if __name__ == "__main__":
	main()