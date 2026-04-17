import os
from typing import Dict

import dotenv
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tiny_sae import Sae, SaeConfig, TrainConfig, train_sae

from preprocesamiento import cargar_comentarios


dotenv.load_dotenv()

# CONFIGURACIÓN GENERAL

MODEL = "openai-community/gpt2"  # puedes cambiar a otro modelo causal
CONTEXT_LEN = int(os.getenv("SAE_CONTEXT_LEN", "512"))
DEVICE = os.getenv("SAE_DEVICE", "cuda:1" if torch.cuda.is_available() else "cpu")
HOOKPOINT = os.getenv("SAE_HOOKPOINT", "transformer.h.8")

# Ajustes de rendimiento (sobre-escribibles por variables de entorno)
TOKENIZE_BATCH_SIZE = int(os.getenv("SAE_TOKENIZE_BATCH_SIZE", "256"))
TOKENIZE_NUM_PROC = int(
	os.getenv("SAE_TOKENIZE_NUM_PROC", str(min(16, max(1, (os.cpu_count() or 4) - 1))))
)
MODEL_BATCH_SIZE = int(os.getenv("SAE_MODEL_BATCH_SIZE", "0"))  # 0 = auto
TARGET_VRAM_UTIL = float(os.getenv("SAE_TARGET_VRAM_UTIL", "0.95"))
SAVE_REPR_EVERY_N_STEPS = int(os.getenv("SAE_SAVE_REPR_EVERY_N_STEPS", "1000"))
SAVE_EVERY_N_TOKENS = int(os.getenv("SAE_SAVE_EVERY_N_TOKENS", "10000000"))
OPTIMIZE_EVERY_N_TOKENS = int(os.getenv("SAE_OPTIMIZE_EVERY_N_TOKENS", "8192"))

# Directorio para checkpoints (en /hdd para no ocupar /home)
CHECKPOINT_DIR = os.getenv(
	"SAE_CHECKPOINT_DIR", "/hdd/aitziber.l/TFM/sae-ckpts/sae-gpt2-comments"
)

# Rutas a tus datos
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"

# Columna de texto en el CSV de comentarios
TEXT_COLUMN = "body"  # en tu CSV hemos visto que es 'body'


# Limitar nº de comentarios para pruebas (None = todos)
MAX_COMMENTS = None  # p.ej. 200_000 para un subset

def configurar_rendimiento_cuda() -> None:
	"""Activa optimizaciones de matmul en GPUs NVIDIA modernas."""
	if not torch.cuda.is_available():
		return

	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True
	torch.set_float32_matmul_precision("high")


def detectar_batch_size_optimo(
	model,
	hookpoint: str,
	context_len: int,
	target_vram_util: float = 0.95,
) -> int:
	"""Busca un batch size alto sin OOM, aproximando la ocupación objetivo de VRAM.

	Se usa el mismo truco del entrenamiento: se corta el forward en el hookpoint,
	así la estimación se parece al coste real de train_sae.
	"""

	if MODEL_BATCH_SIZE > 0:
		print(f"Usando SAE_MODEL_BATCH_SIZE forzado por entorno: {MODEL_BATCH_SIZE}")
		return MODEL_BATCH_SIZE

	if not torch.cuda.is_available():
		return 8

	device = next(model.parameters()).device
	device_index = device.index if device.index is not None else 0
	total_mem = torch.cuda.get_device_properties(device_index).total_memory
	vocab_size = int(getattr(model.config, "vocab_size", 50_257))

	stop_exc = StopIteration("stop_at_hook")

	def _stop_hook(module, inputs, outputs):
		raise stop_exc

	handle = model.get_submodule(hookpoint).register_forward_hook(_stop_hook)

	def _try_batch(bs: int):
		torch.cuda.empty_cache()
		torch.cuda.reset_peak_memory_stats(device_index)
		input_ids = torch.randint(
			low=0,
			high=vocab_size,
			size=(bs, context_len),
			device=device,
			dtype=torch.long,
		)
		try:
			with torch.inference_mode():
				model(input_ids)
		except StopIteration as e:
			if str(e) != str(stop_exc):
				raise
		finally:
			del input_ids
			torch.cuda.synchronize(device_index)

		peak = torch.cuda.max_memory_allocated(device_index)
		util = peak / total_mem
		return util, peak

	# El forward-only subestima ~3x la VRAM real (backward necesita gradientes
	# + estados del optimizer). Escalamos el target para compensar.
	fwd_target = target_vram_util / 3.0

	try:
		lo_ok = 8
		hi = lo_ok
		best_util = 0.0

		while hi <= 2048:
			try:
				util, peak = _try_batch(hi)
				lo_ok = hi
				best_util = util
				print(
					f"[autotune] batch={hi} | pico VRAM(fwd)={peak / 1024**3:.2f} GiB "
					f"({util * 100:.1f}%) | est. train ~{util * 3 * 100:.0f}%"
				)
				if util >= fwd_target:
					return hi
				hi *= 2
			except RuntimeError as e:
				if "out of memory" not in str(e).lower():
					raise
				torch.cuda.empty_cache()
				break

		left = lo_ok
		right = max(lo_ok + 1, hi - 1)

		while left <= right:
			mid = (left + right) // 2
			try:
				util, peak = _try_batch(mid)
				lo_ok = mid
				best_util = util
				print(
					f"[autotune] batch={mid} | pico VRAM(fwd)={peak / 1024**3:.2f} GiB "
					f"({util * 100:.1f}%) | est. train ~{util * 3 * 100:.0f}%"
				)
				if util >= fwd_target:
					return mid
				left = mid + 1
			except RuntimeError as e:
				if "out of memory" not in str(e).lower():
					raise
				torch.cuda.empty_cache()
				right = mid - 1

		print(
			f"[autotune] batch final={lo_ok} "
			f"(fwd {best_util * 100:.1f}%, est. train ~{best_util * 3 * 100:.0f}%)"
		)
		return max(1, lo_ok)
	finally:
		handle.remove()
		torch.cuda.empty_cache()


# CARGA Y PREPARACIÓN DE DATOS


def cargar_dataset_texto() -> Dataset:
	"""Carga TODOS los comentarios y prepara un Dataset solo con el texto.

	La SAE es un modelo no supervisado, así que no necesita
	etiquetas; se entrena sobre todo el corpus de comentarios para
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


# TOKENIZACIÓN Y MODELO


def preparar_modelo_y_datos(dataset: Dataset):
	"""Prepara el tokenizer, el modelo y tokeniza el dataset."""

	configurar_rendimiento_cuda()

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

	# Ruta donde se guarda/carga el dataset tokenizado para no repetir el mapeo
	cache_path = "data/tokenized_gpt2"

	if os.path.isdir(cache_path):
		print(f"Cargando dataset tokenizado desde {cache_path} ...")
		tokenized = Dataset.load_from_disk(cache_path)
	else:
		print("Tokenizando dataset (primera vez, se guardará en disco)...")
		tokenized = dataset.map(
			_tokenize_fn,
			batched=True,
			batch_size=TOKENIZE_BATCH_SIZE,
			num_proc=TOKENIZE_NUM_PROC,
			load_from_cache_file=True,
		)
		tokenized.save_to_disk(cache_path)
		print(f"Dataset tokenizado guardado en {cache_path}")

	return tokenizer, gpt, tokenized


# CONFIGURACIÓN Y ENTRENAMIENTO DE LA SAE


def entrenar_sae(dataset: Dataset):
	"""Entrena una SAE sobre las activaciones de GPT2 usando tus textos."""

	_, gpt, tokenized = preparar_modelo_y_datos(dataset)

	print("Configurando SAE...")
	hidden_size = getattr(gpt.config, "hidden_size", None)
	if hidden_size is None:
		hidden_size = getattr(gpt.config, "n_embd", None)
	if hidden_size is None:
		raise ValueError("No se pudo determinar la dimensión oculta (hidden_size) de GPT-2.")

	print(f"Dimensión oculta detectada para GPT-2: d_in={hidden_size}")

	sae_cfg = SaeConfig(
		d_in=hidden_size,
		num_latents=2**14,
		k=64,
		hookpoint=HOOKPOINT,
	)

	sae = Sae(sae_cfg, device=DEVICE)

	batch_size = detectar_batch_size_optimo(
		model=gpt,
		hookpoint=sae_cfg.hookpoint,
		context_len=CONTEXT_LEN,
		target_vram_util=TARGET_VRAM_UTIL,
	)
	print(f"Batch size usado para entrenamiento SAE: {batch_size}")

	# Checkpoint cada 20% del dataset
	total_tokens = len(tokenized) * CONTEXT_LEN
	save_every = max(1, total_tokens // 5)
	print(f"Total tokens: {total_tokens:,} | checkpoint cada 20% = {save_every:,} tokens")

	# Validar que el directorio de checkpoints es escribible
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
	_test_file = os.path.join(CHECKPOINT_DIR, ".write_test")
	try:
		with open(_test_file, "w") as f:
			f.write("ok")
		os.remove(_test_file)
	except OSError as e:
		raise RuntimeError(
			f"No se puede escribir en {CHECKPOINT_DIR}. "
			"Pide al admin: sudo mkdir -p /hdd/aitziber.l && "
			"sudo chown aitziber.l:aitziber.l /hdd/aitziber.l"
		) from e
	print(f"Checkpoints se guardar\u00e1n en: {CHECKPOINT_DIR}")

	print("Configurando entrenamiento de la SAE...")
	train_cfg = TrainConfig(
		wandb_project="tiny-sae-comments",
		wandb_name="sae-gpt2-comments",
		save_every_n_tokens=save_every,
		optimize_every_n_tokens=OPTIMIZE_EVERY_N_TOKENS,
		model_batch_size=batch_size,
		mask_first_n_tokens=1,
		save_repr_every_n_steps=SAVE_REPR_EVERY_N_STEPS,
		checkpoint_dir=CHECKPOINT_DIR,
	)

	print(f"Representaciones SAE/GPT guardadas cada {SAVE_REPR_EVERY_N_STEPS} steps")
	print(f"Checkpoint SAE guardado cada {SAVE_EVERY_N_TOKENS} tokens")

	print("Iniciando entrenamiento de la SAE...")
	train_sae(
		sae=sae,
		model=gpt,
		token_iterator=tokenized,
		train_cfg=train_cfg,
		use_wandb=False,
	)

	# Guardamos la SAE en disco para usarla luego como extractor de características
	output_dir = CHECKPOINT_DIR
	os.makedirs(output_dir, exist_ok=True)
	print(f"Guardando SAE final en {output_dir} ...")
	sae.save_to_disk(output_dir)

	print("Entrenamiento de SAE finalizado.")


if __name__ == "__main__":
	print("Cargando dataset de todos los comentarios...")
	ds = cargar_dataset_texto()

	print("Tamaño del dataset (todos los comentarios):", len(ds))

	# Entrena la SAE sobre tus textos
	entrenar_sae(ds)

	print("\nA continuación, el siguiente paso será usar esta SAE para extraer\n"
	      "características de los textos (códigos latentes) y entrenar un\n"
	      "clasificador supervisado de género (por ejemplo, una regresión\n"
	      "logística o una pequeña red neuronal) usando esas características.")
