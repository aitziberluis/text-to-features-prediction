"""Extraccion CACHEADA y MULTI-GPU de activaciones de Qwen (capa intermedia).

Resuelve el cuello de botella del proyecto: cada clasificador de Qwen repetia
el forward pass completo (4B params, 26 capas) sobre 5.5M-15.6M comentarios en
CADA ejecucion (~7-22 h por dataset en 1 GPU). Aqui:

  * Se extraen las activaciones del hookpoint UNA sola vez y se cachean en /hdd
    como memmaps float16 (last_token + mean). Reejecuciones = carga instantanea.
  * La extraccion se reparte entre VARIAS GPUs (un proceso worker por GPU,
    shards disjuntos), escalando ~linealmente.
  * Cache compartido para MBTI: las 4 dimensiones son ~99.9% los mismos
    comentarios, asi que se extrae UN canonico y cada rasgo indexa con
    `comment_id` (posicion estable en la lista de comentarios limpia).

Uso como modulo (desde los clasificadores):

    from extraer_qwen import cargar_o_extraer_cache, CacheActivaciones
    cache = cargar_o_extraer_cache(cache_dir, df, gpus=[0,1,2,3,4,5])
    last_token, mean_token = cache.last_token, cache.mean_token   # memmaps (n, H)
    # para una vista (p.ej. un rasgo MBTI sobre el cache canonico):
    idx = cache.indices_para(df_rasgo["comment_id"].to_numpy())

Uso como worker (interno, lo lanza el driver):

    CUDA_VISIBLE_DEVICES=<gpu> python extraer_qwen.py --worker \
        --cache-dir DIR --shard i --nshards N --batch-size 256

El driver (parent) no toca CUDA: prepara dataset.parquet, pre-asigna los
memmaps y lanza los workers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL = "Qwen/Qwen3-4B-Base"
HOOKPOINT = "model.layers.25"
CONTEXT_LEN = 256
HIDDEN_SIZE = 2560
STORAGE_DTYPE = np.float16

# nombres de ficheros dentro de cada cache_dir
F_LAST = "last_token.f16.mmap"
F_MEAN = "mean_token.f16.mmap"
F_META = "meta.json"
F_DATASET = "dataset.parquet"          # text + comment_id (+ author) en orden de cache
F_COMMENT_ID = "comment_id.npy"        # int64, comment_id por fila (orden de cache)
F_SHARD_DONE = "_shard_{i:03d}_of_{n:03d}.done"


# ----------------------------------------------------------------------------
# utilidades de cache
# ----------------------------------------------------------------------------
def _meta_actual(n: int) -> Dict[str, object]:
    return {
        "model": MODEL,
        "hookpoint": HOOKPOINT,
        "context_len": CONTEXT_LEN,
        "hidden_size": HIDDEN_SIZE,
        "storage_dtype": "float16",
        "n": int(n),
    }


def _meta_compatible(meta: Dict[str, object], n: int) -> bool:
    ref = _meta_actual(n)
    return all(meta.get(k) == v for k, v in ref.items())


def _shard_done_path(cache_dir: str, i: int, n: int) -> str:
    return os.path.join(cache_dir, F_SHARD_DONE.format(i=i, n=n))


def _todos_los_shards_hechos(cache_dir: str, nshards: int) -> bool:
    return all(os.path.exists(_shard_done_path(cache_dir, i, nshards)) for i in range(nshards))


# Cada fila ocupa HIDDEN_SIZE*2 = 5120 bytes (fp16). La pagina del SO es 4096.
# Si dos workers escriben filas adyacentes que comparten pagina -> condicion de
# carrera al hacer flush (corrupcion). Como 5120*4 = 20480 = 5*4096, alinear las
# fronteras de shard a multiplos de 4 filas hace que cada shard empiece en una
# frontera de pagina => paginas DISJUNTAS entre workers, sin carreras.
_ROWS_ALIGN = 4


def _shard_bounds(n: int, nshards: int) -> List[int]:
    """Fronteras [b0=0, b1, ..., bN=n] con las interiores alineadas a multiplo de 4 filas."""
    bounds = [0]
    for k in range(1, nshards):
        b = (k * n) // nshards
        b = (b // _ROWS_ALIGN) * _ROWS_ALIGN  # alinear hacia abajo a multiplo de 4
        b = max(bounds[-1], min(b, n))
        bounds.append(b)
    bounds.append(n)
    return bounds


class VistaActivaciones:
    """Array virtual que reordena un memmap del cache al orden de una tarea.

    Sustituto "drop-in" de un np.ndarray (n_tarea, H) para los clasificadores:
    soporta indexado por arrays de enteros y mascaras booleanas, devolviendo
    siempre un ndarray materializado (como hacia el codigo original con
    `np.asarray(feats[idx])`). `pos` mapea fila-de-tarea -> fila-del-cache.
    """

    def __init__(self, memmap: np.memmap, pos: np.ndarray):
        self._mm = memmap
        self._pos = np.asarray(pos, dtype=np.int64)
        self.shape = (len(self._pos), memmap.shape[1])
        self.dtype = memmap.dtype

    def __len__(self) -> int:
        return len(self._pos)

    def __getitem__(self, key):
        key = np.asarray(key)
        rows = self._pos[key]                      # soporta int-array y bool-mask
        return np.asarray(self._mm[rows])


class CacheActivaciones:
    """Vista de solo-lectura sobre un cache de activaciones en disco."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        with open(os.path.join(cache_dir, F_META)) as f:
            self.meta = json.load(f)
        self.n = int(self.meta["n"])
        self.hidden_size = int(self.meta["hidden_size"])
        self.last_token = np.memmap(
            os.path.join(cache_dir, F_LAST), dtype=STORAGE_DTYPE, mode="r",
            shape=(self.n, self.hidden_size),
        )
        self.mean_token = np.memmap(
            os.path.join(cache_dir, F_MEAN), dtype=STORAGE_DTYPE, mode="r",
            shape=(self.n, self.hidden_size),
        )
        self.comment_id = np.load(os.path.join(cache_dir, F_COMMENT_ID))
        # mapa comment_id -> fila en el cache (para vistas)
        self._id2row: Optional[Dict[int, int]] = None

    def indices_para(self, comment_ids: np.ndarray) -> np.ndarray:
        """Devuelve las filas del cache correspondientes a `comment_ids`.

        Lanza si algun comment_id no esta en el cache (el cache debe ser
        un superconjunto de la vista pedida).
        """
        if self._id2row is None:
            self._id2row = {int(c): i for i, c in enumerate(self.comment_id)}
        try:
            return np.fromiter((self._id2row[int(c)] for c in comment_ids),
                               dtype=np.int64, count=len(comment_ids))
        except KeyError as e:
            raise KeyError(
                f"comment_id {e} no esta en el cache {self.cache_dir}. "
                f"El cache no cubre la vista solicitada."
            )


# ----------------------------------------------------------------------------
# DRIVER (parent, sin CUDA): prepara dataset, lanza workers, valida
# ----------------------------------------------------------------------------
def cargar_o_extraer_cache(
    cache_dir: str,
    df: pd.DataFrame,
    gpus: Optional[Sequence[int]] = None,
    batch_size: int = 256,
    num_tok_workers: int = 8,
    forzar: bool = False,
) -> CacheActivaciones:
    """Devuelve un CacheActivaciones, extrayendo si hace falta.

    `df` debe tener columnas 'text' y 'comment_id'. (Si trae 'author' se guarda.)
    `gpus`: lista de ids fisicos a usar; si None, autodetecta GPUs libres.
    """
    n = len(df)
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, F_META)

    # cache valido y completo?
    if not forzar and os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if _meta_compatible(meta, n):
                cache = CacheActivaciones(cache_dir)
                # validar que los comment_id coinciden con los pedidos
                if np.array_equal(cache.comment_id, df["comment_id"].to_numpy()):
                    print(f"[extraer_qwen] Cache valido en {cache_dir} ({n:,} filas). Cargando.", flush=True)
                    return cache
                print("[extraer_qwen] comment_id del cache no coincide; re-extrayendo.", flush=True)
            else:
                print("[extraer_qwen] meta incompatible; re-extrayendo.", flush=True)
        except Exception as e:
            print(f"[extraer_qwen] cache ilegible ({e}); re-extrayendo.", flush=True)

    if gpus is None:
        gpus = detectar_gpus_libres()
    if not gpus:
        raise RuntimeError("No hay GPUs disponibles para la extraccion.")
    nshards = len(gpus)
    print(f"[extraer_qwen] Extrayendo {n:,} comentarios en {nshards} GPU(s): {list(gpus)}", flush=True)

    # 1) guardar dataset.parquet (text, comment_id, [author]) en orden de cache
    cols = ["comment_id", "text"]
    if "author" in df.columns:
        cols.append("author")
    ds = df[cols].reset_index(drop=True)
    ds.to_parquet(os.path.join(cache_dir, F_DATASET), index=False)
    np.save(os.path.join(cache_dir, F_COMMENT_ID), ds["comment_id"].to_numpy().astype(np.int64))

    # 2) pre-asignar memmaps float16 (ficheros dispersos del tamano total)
    for fname in (F_LAST, F_MEAN):
        mm = np.memmap(os.path.join(cache_dir, fname), dtype=STORAGE_DTYPE,
                       mode="w+", shape=(n, HIDDEN_SIZE))
        del mm  # crea/trunca el fichero al tamano correcto

    # 3) lanzar un worker por GPU (shards estriados balanceados por longitud)
    procs = []
    for i, gpu in enumerate(gpus):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TOKENIZERS_PARALLELISM"] = "false"
        # Reduce memory fragmentation on CUDA and allow expandable segments
        # so workers loading large models are less likely to OOM immediately.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        cmd = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--cache-dir", cache_dir,
            "--shard", str(i), "--nshards", str(nshards),
            "--batch-size", str(batch_size),
            "--num-tok-workers", str(num_tok_workers),
        ]
        print(f"[extraer_qwen]   -> worker shard {i}/{nshards} en GPU fisica {gpu}", flush=True)
        # Stagger worker starts to avoid all processes loading the model
        # simultaneously which spikes GPU memory usage.
        try:
            time.sleep(i * 3)
        except Exception:
            pass
        procs.append(subprocess.Popen(cmd, env=env))

    # 4) esperar
    fallos = 0
    for i, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            fallos += 1
            print(f"[extraer_qwen] worker shard {i} fallo con codigo {rc}", flush=True)
    if fallos:
        raise RuntimeError(f"{fallos} worker(s) fallaron; cache incompleto en {cache_dir}")
    if not _todos_los_shards_hechos(cache_dir, nshards):
        raise RuntimeError(f"Faltan shards en {cache_dir} pese a rc=0")

    # 5) escribir meta y limpiar marcadores
    with open(meta_path, "w") as f:
        json.dump(_meta_actual(n), f, indent=2)
    for i in range(nshards):
        try:
            os.remove(_shard_done_path(cache_dir, i, nshards))
        except OSError:
            pass
    print(f"[extraer_qwen] Cache completo en {cache_dir}", flush=True)
    return CacheActivaciones(cache_dir)


def detectar_gpus_libres(umbral_mem_mib: int = 2000, umbral_util: int = 30) -> List[int]:
    """Devuelve ids de GPUs con poca memoria usada y baja utilizacion."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True,
        )
    except Exception as e:
        print(f"[extraer_qwen] nvidia-smi no disponible ({e}); usando GPU 0", flush=True)
        return [0]
    libres = []
    for line in out.strip().splitlines():
        idx, mem, util = (x.strip() for x in line.split(","))
        if int(mem) <= umbral_mem_mib and int(util) <= umbral_util:
            libres.append(int(idx))
    return libres


# ----------------------------------------------------------------------------
# WORKER (con CUDA): extrae su shard y escribe en los memmaps compartidos
# ----------------------------------------------------------------------------
def _ejecutar_worker(cache_dir: str, shard: int, nshards: int,
                     batch_size: int, num_tok_workers: int) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer

    # parche defensivo: algunas instalaciones rompen el import perezoso de
    # quantizers por torchao incompatible; no afecta a la inferencia fp16.
    try:
        import torchao.quantization as _q  # noqa
        for _n in ("Float8WeightOnlyConfig", "Float8DynamicActivationFloat8WeightConfig",
                   "Int4WeightOnlyConfig", "Int8WeightOnlyConfig",
                   "Int8DynamicActivationInt8WeightConfig", "FbgemmFp8Config",
                   "ModuleFqnToConfig", "Float8Tensor"):
            if not hasattr(_q, _n):
                setattr(_q, _n, type(_n, (), {}))
    except Exception:
        pass

    from transformers import AutoModelForCausalLM
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _clasificador_utils_qwen import truncar_modelo_para_hookpoint

    t0 = time.time()
    ds = pd.read_parquet(os.path.join(cache_dir, F_DATASET))
    n = len(ds)
    textos_all = ds["text"].astype(str).tolist()

    # Shards CONTIGUOS con fronteras alineadas a pagina (sin carreras de escritura).
    # El CSV no esta ordenado por longitud, asi que trozos contiguos estan
    # naturalmente balanceados. Cada worker reordena SU trozo por longitud para
    # bucketing (escribe en filas absolutas dentro de su rango, sin solapar paginas).
    char_lens = np.fromiter((len(t) for t in textos_all), dtype=np.int64, count=n)
    bounds = _shard_bounds(n, nshards)
    start, end = bounds[shard], bounds[shard + 1]
    mias = np.arange(start, end, dtype=np.int64)
    mias = mias[np.argsort(char_lens[mias], kind="stable")]
    print(f"[worker {shard}/{nshards}] {len(mias):,}/{n:,} filas; setup {time.time()-t0:.1f}s", flush=True)

    device = "cuda:0"  # CUDA_VISIBLE_DEVICES ya mapea la GPU fisica
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa", device_map={"": device},
    )
    model.eval()
    trunk, _, keep = truncar_modelo_para_hookpoint(model, HOOKPOINT)
    model._trunk = trunk
    hookpoint_module = model.get_submodule(HOOKPOINT)
    captured = {}

    def hook(_m, _i, o):
        captured["a"] = (o[0] if isinstance(o, tuple) else o).detach()

    handle = hookpoint_module.register_forward_hook(hook)
    print(f"[worker {shard}/{nshards}] modelo cargado ({keep} capas) {time.time()-t0:.1f}s", flush=True)

    last_mm = np.memmap(os.path.join(cache_dir, F_LAST), dtype=STORAGE_DTYPE, mode="r+",
                        shape=(n, HIDDEN_SIZE))
    mean_mm = np.memmap(os.path.join(cache_dir, F_MEAN), dtype=STORAGE_DTYPE, mode="r+",
                        shape=(n, HIDDEN_SIZE))

    class _DS(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, k):
            r = int(self.rows[k])
            return textos_all[r], r

    def _collate(batch):
        txts = [b[0] for b in batch]
        rows = [b[1] for b in batch]
        tok = tokenizer(txts, max_length=CONTEXT_LEN, truncation=True, padding=True,
                        return_attention_mask=True, return_tensors="pt")
        return tok["input_ids"], tok["attention_mask"], np.asarray(rows, dtype=np.int64)

    loader = DataLoader(_DS(mias), batch_size=batch_size, num_workers=num_tok_workers,
                        collate_fn=_collate, pin_memory=True, shuffle=False,
                        persistent_workers=False)

    total = len(mias)
    done = 0
    last_print = time.time()
    dev = torch.device(device)
    with torch.inference_mode():
        for input_ids, attn, rows in loader:
            input_ids = input_ids.to(dev, non_blocking=True)
            attn = attn.to(dev, non_blocking=True)
            model._trunk(input_ids=input_ids, attention_mask=attn)
            acts = captured["a"].float()
            mask = attn.float()
            lengths = mask.sum(1).clamp(min=1).long() - 1
            bidx = torch.arange(acts.shape[0], device=dev)
            last_tok = acts[bidx, lengths, :]
            denom = mask.sum(1, keepdim=True).clamp(min=1)
            mean_tok = (acts * mask.unsqueeze(-1)).sum(1) / denom
            last_np = np.nan_to_num(last_tok.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
            mean_np = np.nan_to_num(mean_tok.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)
            last_mm[rows] = last_np.astype(STORAGE_DTYPE)
            mean_mm[rows] = mean_np.astype(STORAGE_DTYPE)
            done += len(rows)
            now = time.time()
            if now - last_print >= 60 or done >= total:
                cps = done / max(1e-9, now - t0)
                eta = (total - done) / max(1e-9, cps)
                print(f"[worker {shard}/{nshards}] {100*done/total:5.1f}% "
                      f"({done:,}/{total:,}) {cps:.0f} c/s ETA {eta/3600:.1f}h "
                      f"seqlen={input_ids.shape[1]}", flush=True)
                last_print = now

    handle.remove()
    last_mm.flush(); mean_mm.flush()
    del last_mm, mean_mm
    Path(_shard_done_path(cache_dir, shard, nshards)).touch()
    print(f"[worker {shard}/{nshards}] HECHO en {(time.time()-t0)/3600:.2f}h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-tok-workers", type=int, default=8)
    args = ap.parse_args()
    if not args.worker:
        raise SystemExit("Este script se usa como modulo (cargar_o_extraer_cache) "
                         "o como worker (--worker). Ver docstring.")
    _ejecutar_worker(args.cache_dir, args.shard, args.nshards,
                     args.batch_size, args.num_tok_workers)


if __name__ == "__main__":
    main()
