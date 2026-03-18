import os
from typing import Dict

import dotenv
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tiny_sae import Sae, SaeConfig, TrainConfig, train_sae

from preprocesamiento import cargar_comentarios


dotenv.load_dotenv()


# =====================
# CONFIGURACIÓN GENERAL
# =====================

MODEL = "Qwen/Qwen3.5-2B"  # modelo base Qwen
CONTEXT_LEN = int(os.getenv("SAE_CONTEXT_LEN", "512"))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
HOOKPOINT = os.getenv("SAE_HOOKPOINT", "model.layers.15")

# Ajustes de rendimiento (sobre-escribibles por variables de entorno)
TOKENIZE_BATCH_SIZE = int(os.getenv("SAE_TOKENIZE_BATCH_SIZE", "256"))
TOKENIZE_NUM_PROC = int(
    os.getenv("SAE_TOKENIZE_NUM_PROC", str(min(16, max(1, (os.cpu_count() or 4) - 1))))
)
MODEL_BATCH_SIZE = int(os.getenv("SAE_MODEL_BATCH_SIZE", "0"))  # 0 = auto
TARGET_VRAM_UTIL = float(os.getenv("SAE_TARGET_VRAM_UTIL", "0.90"))
SAVE_REPR_EVERY_N_STEPS = int(os.getenv("SAE_SAVE_REPR_EVERY_N_STEPS", "1000"))
USE_FLASH_ATTN = os.getenv("SAE_FLASH_ATTN", "1") == "1"

# Rutas a tus datos
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"

# Columna de texto en el CSV de comentarios
TEXT_COLUMN = "body"

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
    target_vram_util: float = 0.90,
) -> int:
    """Busca un batch size alto sin OOM, aproximando la ocupación objetivo de VRAM.

    Se usa el mismo truco del entrenamiento: se corta el forward en el hookpoint,
    así la estimación se parece al coste real de `train_sae`.
    """

    if MODEL_BATCH_SIZE > 0:
        print(f"Usando SAE_MODEL_BATCH_SIZE forzado por entorno: {MODEL_BATCH_SIZE}")
        return MODEL_BATCH_SIZE

    if not torch.cuda.is_available():
        return 8

    device = next(model.parameters()).device
    device_index = device.index if device.index is not None else 0
    total_mem = torch.cuda.get_device_properties(device_index).total_memory
    vocab_size = int(getattr(model.config, "vocab_size", 151_936))

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
                    f"[autotune] batch={hi} | pico VRAM={peak / 1024**3:.2f} GiB "
                    f"({util * 100:.1f}%)"
                )
                if util >= target_vram_util:
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
                    f"[autotune] batch={mid} | pico VRAM={peak / 1024**3:.2f} GiB "
                    f"({util * 100:.1f}%)"
                )
                if util >= target_vram_util:
                    return mid
                left = mid + 1
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                right = mid - 1

        print(
            f"[autotune] batch final={lo_ok} "
            f"(ocupación estimada {best_util * 100:.1f}%)"
        )
        return max(1, lo_ok)
    finally:
        handle.remove()
        torch.cuda.empty_cache()


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

    configurar_rendimiento_cuda()

    print("Cargando tokenizer y modelo Qwen...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": {"": DEVICE},
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available() and USE_FLASH_ATTN:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, **model_kwargs)
    except Exception as e:
        if "attn_implementation" in model_kwargs:
            print(f"FlashAttention no disponible ({e}). Reintentando sin flash_attention_2...")
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(MODEL, **model_kwargs)
        else:
            raise

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
        batch_size=TOKENIZE_BATCH_SIZE,
        num_proc=TOKENIZE_NUM_PROC,
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
        hookpoint=HOOKPOINT,
    )

    sae = Sae(sae_cfg, device=DEVICE)

    batch_size = detectar_batch_size_optimo(
        model=model,
        hookpoint=sae_cfg.hookpoint,
        context_len=CONTEXT_LEN,
        target_vram_util=TARGET_VRAM_UTIL,
    )
    print(f"Batch size usado para entrenamiento SAE: {batch_size}")

    print("Configurando entrenamiento de la SAE...")
    train_cfg = TrainConfig(
        wandb_project="tiny-sae-qwen",
        wandb_name="sae-qwen3.5-2b",
        save_every_n_tokens=10_000_000,
        optimize_every_n_tokens=8192,
        model_batch_size=batch_size,
        mask_first_n_tokens=1,
        save_repr_every_n_steps=SAVE_REPR_EVERY_N_STEPS,
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
