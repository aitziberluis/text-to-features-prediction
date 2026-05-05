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
MODEL = os.getenv("SAE_MODEL", "Qwen/Qwen3-4B-Base")  # 36 capas, hidden=2560
CONTEXT_LEN = int(os.getenv("SAE_CONTEXT_LEN", "256"))  # P99 token len ~391; 2.5% truncados
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Capa elegida por layer-sweep (find_best_layer.py): la 25 ha sido la mejor
# capa promedio en F1 macro normalizado (score=0.9925) sobre genero/edad/MBTI.
HOOKPOINT = os.getenv("SAE_HOOKPOINT", "model.layers.25")
# Ajustes de rendimiento (sobre-escribibles por variables de entorno)
TOKENIZE_BATCH_SIZE = int(os.getenv("SAE_TOKENIZE_BATCH_SIZE", "256"))
TOKENIZE_NUM_PROC = int(os.getenv("SAE_TOKENIZE_NUM_PROC", str(min(16, max(1, (os.cpu_count() or 4) - 1)))))
MODEL_BATCH_SIZE = int(os.getenv("SAE_MODEL_BATCH_SIZE", "0"))  # 0 = auto
TARGET_VRAM_UTIL = float(os.getenv("SAE_TARGET_VRAM_UTIL", "0.90"))
SAVE_REPR_EVERY_N_STEPS = int(os.getenv("SAE_SAVE_REPR_EVERY_N_STEPS", "1000"))
USE_FLASH_ATTN = os.getenv("SAE_FLASH_ATTN", "1") == "1"
#Directorio para checkpoints (en /hdd para no ocupar /home)
#Por defecto se usa directamente la carpeta /hdd/aitziber.l que has creado.
CHECKPOINT_DIR = os.getenv("SAE_CHECKPOINT_DIR", "/hdd/aitziber.l")
#Limite de tokens de entrenamiento (0 = sin limite, usa todo el dataset).
#Para una SAE de 16k latentes ~1B tokens es mas que suficiente.
MAX_TOKENS = int(os.getenv("SAE_MAX_TOKENS", "0"))
# Rutas a tus datos
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
# Columna de texto en el CSV de comentarios
TEXT_COLUMN = "body"
#Limitar nº de comentarios para pruebas (None = todos)
MAX_COMMENTS = None  #p.ej. 200_000 para un subset
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
    target_vram_util: float = 0.75,
) -> int:
    """Busca un batch size alto sin OOM, aproximando la ocupación objetivo de VRAM.
    Se usa el mismo truco del entrenamiento: se corta el forward en el hookpoint, así la estimación se parece al coste real de `train_sae`.
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
        input_ids = torch.randint( low=0, high=vocab_size, size=(bs, context_len), device=device, dtype=torch.long,
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
    #El forward-only subestima ~4.5x la VRAM real (backward necesita gradientes de las activaciones del SAE + estados del optimizer Adam fp32).
    #Empiricamente: batch 128 con fwd ~32% peto en train (>100%).
    SAE_TRAIN_FACTOR = 4.5
    fwd_target = target_vram_util / SAE_TRAIN_FACTOR

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
                    f"({util * 100:.1f}%) | est. train ~{util * SAE_TRAIN_FACTOR * 100:.0f}%"
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
                    f"({util * 100:.1f}%) | est. train ~{util * SAE_TRAIN_FACTOR * 100:.0f}%"
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
            f"(fwd {best_util * 100:.1f}%, est. train ~{best_util * SAE_TRAIN_FACTOR * 100:.0f}%)"
        )
        return max(1, lo_ok)
    finally:
        handle.remove()
        torch.cuda.empty_cache()

# CARGA Y PREPARACIÓN DE DATOS
def cargar_dataset_texto() -> Dataset:
    """Carga TODOS los comentarios y prepara un Dataset solo con el texto.
    La SAE es un modelo no supervisado, así que se entrena sobre todo el corpus de comentarios, sin usar etiquetas de género/edad.
    """
    df_comentarios = cargar_comentarios(PATH_COMENTARIOS, nrows=MAX_COMMENTS)
    if TEXT_COLUMN in df_comentarios.columns and TEXT_COLUMN != "text":
        df_comentarios = df_comentarios.rename(columns={TEXT_COLUMN: "text"})
    df_comentarios = df_comentarios.dropna(subset=["text"]).reset_index(drop=True)
    dataset = Dataset.from_pandas(df_comentarios[["text"]])
    return dataset


# TOKENIZACIÓN Y MODELO
def preparar_modelo_y_datos(dataset: Dataset):
    """Prepara tokenizer, modelo Qwen y tokeniza el dataset."""
    configurar_rendimiento_cuda()
    print("Cargando tokenizer y modelo Qwen...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    # Auto-pick: BF16 nativo en Ampere+ (RTX A6000, sm_86); FP16 en Turing
    # (Quadro RTX 8000, sm_75) que no tiene tensor cores BF16.
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        _train_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        _train_dtype = torch.float32
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": {"": DEVICE},
        "torch_dtype": _train_dtype,
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
    model.eval()
    # OPT #1: truncar el modelo despues del HOOKPOINT (Qwen: model.model.layers).
    # Aunque tiny_sae corta el forward via StopIteration, las capas posteriores
    # siguen ocupando VRAM y bloqueando batches mas grandes. Para Qwen3-4B con
    # hookpoint en la capa 17 de 36 esto libera ~50% del peso del modelo.
    import torch.nn as _nn
    _keep = int(HOOKPOINT.rsplit(".", 1)[1]) + 1
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n_total = len(model.model.layers)
        model.model.layers = _nn.ModuleList(model.model.layers[:_keep])
        # Tambien podemos quitar el norm final y el lm_head: nunca se ejecutan.
        if hasattr(model.model, "norm"):
            del model.model.norm
        print(
            f"  Modelo truncado a las primeras {_keep} capas (de {n_total}); "
            f"liberadas {n_total - _keep} capas + final norm + lm_head"
        )
    else:
        print("  AVISO: arquitectura inesperada, no trunco capas.")
    if hasattr(model, "lm_head"):
        del model.lm_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    #NOTA: torch.compile se aplica DESPUES del autotune del batch size y del benchmark de ETA (en entrenar_sae), porque los hooks que lanzan
    #StopIteration confunden a TorchDynamo y provocan InternalTorchDynamoError.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def _tokenize_fn(x: Dict[str, list]):
        out = tokenizer(x["text"],max_length=CONTEXT_LEN,truncation=True,padding="max_length",return_attention_mask=False,
        )
        return out
    #Ruta donde se guarda/carga el dataset tokenizado para no repetir el mapeo
    cache_path = "/hdd/aitziber.l/tokenized_qwen"
    if os.path.isdir(cache_path):
        print(f"Cargando dataset tokenizado desde {cache_path} ...")
        tokenized = Dataset.load_from_disk(cache_path)
    else:
        print("Tokenizando dataset (primera vez, se guardará en disco)...")
        tokenized = dataset.map(_tokenize_fn,batched=True,batch_size=TOKENIZE_BATCH_SIZE,num_proc=TOKENIZE_NUM_PROC,load_from_cache_file=True,
        )
        tokenized.save_to_disk(cache_path)
        print(f"Dataset tokenizado guardado en {cache_path}")
    return tokenizer, model, tokenized
#CONFIGURACIÓN Y ENTRENAMIENTO DE LA SAE
def entrenar_sae(dataset: Dataset):
    """Entrena una SAE sobre las activaciones de Qwen usando tus textos."""
    _, model, tokenized = preparar_modelo_y_datos(dataset)
    #inferimos la dimensión de las activaciones a partir de la config del modelo
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        #fallback genérico: algunos modelos usan "n_embd" u otros campos
        hidden_size = getattr(model.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("No se pudo determinar la dimensión oculta (hidden_size) del modelo Qwen.")
    print(f"Dimensión oculta detectada para Qwen: d_in={hidden_size}")
    print("Configurando SAE...")
    sae_cfg = SaeConfig(
        d_in=hidden_size,
        num_latents=2**14,
        k=64,
        #para Qwen, un hookpoint típico es una capa intermedia del bloque transformer.
        #ajusta este valor si quieres enganchar otra capa.
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
    #truncar dataset si SAE_MAX_TOKENS > 0
    if MAX_TOKENS > 0:
        max_rows = MAX_TOKENS // CONTEXT_LEN
        if max_rows < len(tokenized):
            print(
                f"Limitando dataset a {MAX_TOKENS:,} tokens "
                f"({max_rows:,} filas de {len(tokenized):,})."
            )
            tokenized = tokenized.select(range(max_rows))
    #Checkpoint cada 20% del dataset para si se para o algo no tener que empezar de cero esto solo poner si hay almacenamiento
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
        wandb_project="",
        wandb_name="",
        save_every_n_tokens=save_every,
        optimize_every_n_tokens=8192,
        model_batch_size=batch_size,
        mask_first_n_tokens=1,
        save_repr_every_n_steps=0,            # desactivar guardado de representaciones
        checkpoint_dir=CHECKPOINT_DIR,
    )
    # Mini-benchmark de ETA: medimos tiempo/batch con unos pocos forwards
    # truncados al hookpoint y extrapolamos al numero total de batches.
    print("Estimando tiempo de entrenamiento (warmup forward + backward)...")
    import time as _time
    n_batches_total = (len(tokenized) + batch_size - 1) // batch_size
    vocab_size = int(getattr(model.config, "vocab_size", 151_936))
    stop_exc = StopIteration("eta_stop")
    def _stop_hook(module, inputs, outputs):
        raise stop_exc
    handle = model.get_submodule(sae_cfg.hookpoint).register_forward_hook(_stop_hook)
    try:
        warmup, measured = 2, 5
        times = []
        for i in range(warmup + measured):
            ids = torch.randint(
                low=0, high=vocab_size,
                size=(batch_size, CONTEXT_LEN),
                device=model.device, dtype=torch.long,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(model.device)
            t0 = _time.time()
            try:
                with torch.inference_mode():
                    model(ids)
            except StopIteration as e:
                if str(e) != str(stop_exc):
                    raise
            # Aprox. coste del paso SAE (forward+backward+opt): factor ~2.5x
            # frente al solo-forward del transformer truncado.
            if torch.cuda.is_available():
                torch.cuda.synchronize(model.device)
            dt = _time.time() - t0
            if i >= warmup:
                times.append(dt)
            del ids
    finally:
        handle.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    sec_per_batch = (sum(times) / len(times)) * 2.5  #factor empirico fwd->train
    eta_sec = sec_per_batch * n_batches_total
    h, rem = divmod(int(eta_sec), 3600)
    m, s = divmod(rem, 60)
    print(
        f"  Batches totales: {n_batches_total:,} | "
        f"~{sec_per_batch:.3f} s/batch (estimado) | "
        f"ETA aprox: {h}h {m}m {s}s")
    #Ahora si activamos torch.compile sobre el bloque transformer truncado,despues de autotune y ETA (los hooks con StopIteration confunden a
    #TorchDynamo). dynamic=False porque las shapes son fijas en train_sae.
    if torch.cuda.is_available():
        try:
            model.model = torch.compile(
                model.model, mode="reduce-overhead", dynamic=False, fullgraph=False)
            print("  torch.compile activado (reduce-overhead, static shapes)")
        except Exception as _ce:
            print(f"torch.compile no disponible: {_ce}")
    print("Iniciando entrenamiento de la SAE (Qwen)...")
    train_sae(sae=sae,model=model,token_iterator=tokenized,train_cfg=train_cfg,use_wandb=False,)
    output_dir = CHECKPOINT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"Guardando SAE final en {output_dir} ...")
    sae.save_to_disk(output_dir)
    print("Entrenamiento de SAE con Qwen finalizado.")
if __name__ == "__main__":
    print("Cargando dataset de comentarios...")
    ds = cargar_dataset_texto()
    print("Tamaño del dataset (comentarios):", len(ds))
    entrenar_sae(ds)
