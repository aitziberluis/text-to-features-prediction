"""Extraccion USER-LEVEL multi-GPU de features de GPT-2 y Qwen (pura + SAE).

Para el pipeline de clasificacion con MLP a nivel de USUARIO. En UN solo forward
pass por (modelo, unidad) acumula, por usuario, las cuatro features:

    mean_of_last        = media sobre comentarios del usuario de last_token (H)
    mean_of_mean        = media sobre comentarios de mean-de-tokens         (H)
    mean_of_last_sae    = idem pero codificando el last_token con la SAE     (L)
    mean_of_mean_sae    = idem con la media de tokens codificados            (L)

Solo se cachea la matriz por-usuario (~MBs), nunca las activaciones por
comentario. Multi-GPU: un worker por GPU sobre shards disjuntos; cada uno
acumula sumas parciales por-usuario y el padre las une (suma) y divide por el
conteo. No hay escritura compartida (cada worker vuelca su propio .npz), asi que
el shard puede estar balanceado por longitud sin condiciones de carrera.

Uso como modulo:
    from extraer_userlevel import cargar_o_extraer_userlevel, CacheUserLevel
    cache = cargar_o_extraer_userlevel("qwen", "mbti", df_canon, gpus=[0,1,2,3,4,5])
    # cache.authors, cache.counts, cache.pure_last, cache.pure_mean,
    # cache.sae_last, cache.sae_mean   (todas alineadas a cache.authors)

Worker (interno): --worker --model qwen --cache-dir DIR --shard i --nshards N
"""
from __future__ import annotations

import argparse
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

# ---- configuracion por modelo ---------------------------------------------
MODELOS = {
    "gpt2": {
        "model": "openai-community/gpt2",
        "hookpoint": "transformer.h.8",
        "sae_path": "sae-ckpts/sae-gpt2-comments",
        "trunk_attr": "transformer",      # se llama model.transformer(...)
    },
    "qwen": {
        "model": "Qwen/Qwen3-4B-Base",
        "hookpoint": "model.layers.25",
        "sae_path": "sae-ckpts/qwen",
        "trunk_attr": "_trunk",           # se llama model._trunk(...)
    },
}
CONTEXT_LEN = 256
CACHE_BASE = os.environ.get("QWEN_CACHE_BASE_UL", "/hdd/aitziber.l/cache_userlevel")

F_META = "meta.json"
F_AUTHORS = "authors.npy"
F_COUNTS = "counts.npy"
F_PURE_LAST = "pure_mean_of_last.npy"
F_PURE_MEAN = "pure_mean_of_mean.npy"
F_SAE_LAST = "sae_mean_of_last.npy"
F_SAE_MEAN = "sae_mean_of_mean.npy"
F_DATASET = "dataset.parquet"
F_SHARD = "_shard_{i:03d}_of_{n:03d}.npz"


class CacheUserLevel:
    """Lectura del cache user-level: features por usuario alineadas a `authors`."""

    def __init__(self, cache_dir: str):
        with open(os.path.join(cache_dir, F_META)) as f:
            self.meta = json.load(f)
        self.cache_dir = cache_dir
        self.authors = np.load(os.path.join(cache_dir, F_AUTHORS), allow_pickle=True)
        self.counts = np.load(os.path.join(cache_dir, F_COUNTS))
        self.pure_last = np.load(os.path.join(cache_dir, F_PURE_LAST))
        self.pure_mean = np.load(os.path.join(cache_dir, F_PURE_MEAN))
        self.sae_last = np.load(os.path.join(cache_dir, F_SAE_LAST))
        self.sae_mean = np.load(os.path.join(cache_dir, F_SAE_MEAN))
        self.hidden_size = int(self.meta["hidden_size"])
        self.num_latents = int(self.meta["num_latents"])

    def features(self, representacion: str, pooling: str) -> np.ndarray:
        """representacion in {'pure','sae'}, pooling in {'last','mean'}."""
        return {
            ("pure", "last"): self.pure_last, ("pure", "mean"): self.pure_mean,
            ("sae", "last"): self.sae_last, ("sae", "mean"): self.sae_mean,
        }[(representacion, pooling)]


def cache_dir_userlevel(model_key: str, unit: str) -> str:
    return os.path.join(CACHE_BASE, model_key, unit)


def cargar_userlevel_si_existe(model_key: str, unit: str) -> Optional["CacheUserLevel"]:
    """Devuelve el cache si su meta.json existe y es legible; si no, None.

    Evita reconstruir el dataframe canonico (recarga del CSV de 5.3 GB) cuando
    el cache ya esta listo (caso comun en reejecuciones y en los 4 rasgos MBTI
    que comparten el cache 'mbti')."""
    cache_dir = cache_dir_userlevel(model_key, unit)
    if not os.path.exists(os.path.join(cache_dir, F_META)):
        return None
    try:
        return CacheUserLevel(cache_dir)
    except Exception:
        return None


# ---- DRIVER (sin CUDA) -----------------------------------------------------
def cargar_o_extraer_userlevel(
    model_key: str, unit: str, df_canon: pd.DataFrame,
    gpus: Optional[Sequence[int]] = None, batch_size: int = 256,
    num_tok_workers: int = 8,
) -> CacheUserLevel:
    if model_key not in MODELOS:
        raise ValueError(f"modelo '{model_key}' no en {list(MODELOS)}")
    cache_dir = cache_dir_userlevel(model_key, unit)
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, F_META)
    n = len(df_canon)

    if os.path.exists(meta_path):
        try:
            cache = CacheUserLevel(cache_dir)
            if cache.meta.get("n_comments") == n and cache.meta.get("model_key") == model_key:
                print(f"[userlevel] cache valido {cache_dir} "
                      f"({len(cache.authors):,} usuarios). Cargando.", flush=True)
                return cache
            print("[userlevel] meta incompatible; re-extrayendo.", flush=True)
        except Exception as e:
            print(f"[userlevel] cache ilegible ({e}); re-extrayendo.", flush=True)

    if gpus is None:
        from extraer_qwen import detectar_gpus_libres
        gpus = detectar_gpus_libres()
    if not gpus:
        raise RuntimeError("No hay GPUs disponibles.")
    nshards = len(gpus)

    # autores (orden de usuario = sorted unique)
    authors = np.sort(df_canon["author"].astype(str).unique())
    print(f"[userlevel] {model_key}/{unit}: {n:,} comentarios, {len(authors):,} usuarios, "
          f"{nshards} GPU(s) {list(gpus)}", flush=True)
    cols = ["author", "text"]
    df_canon[cols].reset_index(drop=True).to_parquet(os.path.join(cache_dir, F_DATASET), index=False)
    np.save(os.path.join(cache_dir, F_AUTHORS), authors)

    # lanzar workers
    procs = []
    for i, gpu in enumerate(gpus):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TOKENIZERS_PARALLELISM"] = "false"
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               "--model", model_key, "--cache-dir", cache_dir,
               "--shard", str(i), "--nshards", str(nshards),
               "--batch-size", str(batch_size), "--num-tok-workers", str(num_tok_workers)]
        print(f"[userlevel]   -> worker {i}/{nshards} GPU {gpu}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    fallos = sum(1 for p in procs if p.wait() != 0)
    if fallos:
        raise RuntimeError(f"{fallos} worker(s) fallaron en {cache_dir}")

    # merge: sumar sumas parciales y dividir por conteo (H y L se infieren del shard)
    nu = len(authors)
    sum_pl = sum_pm = sum_sl = sum_sm = None
    counts = np.zeros(nu, dtype=np.float64)
    L = H = None
    for i in range(nshards):
        d = np.load(os.path.join(cache_dir, F_SHARD.format(i=i, n=nshards)))
        if sum_pl is None:
            H = d["sum_pure_last"].shape[1]; L = d["sum_sae_last"].shape[1]
            sum_pl = np.zeros((nu, H), np.float64); sum_pm = np.zeros((nu, H), np.float64)
            sum_sl = np.zeros((nu, L), np.float64); sum_sm = np.zeros((nu, L), np.float64)
        sum_pl += d["sum_pure_last"]; sum_pm += d["sum_pure_mean"]
        sum_sl += d["sum_sae_last"]; sum_sm += d["sum_sae_mean"]
        counts += d["count"]
    cnt = counts.copy(); cnt[cnt == 0] = 1.0
    cnt = cnt[:, None]
    pure_last = (sum_pl / cnt).astype(np.float32)
    pure_mean = (sum_pm / cnt).astype(np.float32)
    sae_last = (sum_sl / cnt).astype(np.float32)
    sae_mean = (sum_sm / cnt).astype(np.float32)

    np.save(os.path.join(cache_dir, F_COUNTS), counts.astype(np.int64))
    np.save(os.path.join(cache_dir, F_PURE_LAST), pure_last)
    np.save(os.path.join(cache_dir, F_PURE_MEAN), pure_mean)
    np.save(os.path.join(cache_dir, F_SAE_LAST), sae_last)
    np.save(os.path.join(cache_dir, F_SAE_MEAN), sae_mean)
    with open(meta_path, "w") as f:
        json.dump({"model_key": model_key, "model": MODELOS[model_key]["model"],
                   "hookpoint": MODELOS[model_key]["hookpoint"], "unit": unit,
                   "context_len": CONTEXT_LEN, "hidden_size": int(H),
                   "num_latents": int(L), "n_comments": int(n), "n_users": int(nu)}, f, indent=2)
    for i in range(nshards):
        try:
            os.remove(os.path.join(cache_dir, F_SHARD.format(i=i, n=nshards)))
        except OSError:
            pass
    print(f"[userlevel] cache completo {cache_dir} ({nu:,} usuarios, H={H}, L={L})", flush=True)
    return CacheUserLevel(cache_dir)


MODELOS_DIMS: Dict[str, Dict[str, int]] = {}  # opcional; H/L se infieren de los shards


# ---- WORKER (con CUDA) -----------------------------------------------------
def _setup(model_key: str, device: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        import torchao.quantization as _q
        for _n in ("Float8WeightOnlyConfig", "Float8DynamicActivationFloat8WeightConfig",
                   "Int4WeightOnlyConfig", "Int8WeightOnlyConfig",
                   "Int8DynamicActivationInt8WeightConfig", "FbgemmFp8Config",
                   "ModuleFqnToConfig", "Float8Tensor"):
            if not hasattr(_q, _n):
                setattr(_q, _n, type(_n, (), {}))
    except Exception:
        pass
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tiny_sae import Sae

    cfg = MODELOS[model_key]
    tok = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.float16, attn_implementation="sdpa", device_map={"": device})
    model.eval()

    # truncar tras el hookpoint
    if model_key == "qwen":
        from _clasificador_utils_qwen import truncar_modelo_para_hookpoint
        trunk, _, keep = truncar_modelo_para_hookpoint(model, cfg["hookpoint"])
        model._trunk = trunk
        def run(ii, am):
            model._trunk(input_ids=ii, attention_mask=am)
    else:  # gpt2
        keep = int(cfg["hookpoint"].rsplit(".", 1)[1]) + 1
        import torch.nn as nn
        model.transformer.h = nn.ModuleList(model.transformer.h[:keep])
        def run(ii, am):
            model.transformer(input_ids=ii, attention_mask=am)

    hook_mod = model.get_submodule(cfg["hookpoint"])
    captured = {}
    hook_mod.register_forward_hook(
        lambda m, i, o: captured.__setitem__("a", (o[0] if isinstance(o, tuple) else o).detach()))

    sae = Sae.load_from_disk(cfg["sae_path"], device=device)
    sae = sae.to(device=device, dtype=torch.float16)
    return tok, model, run, captured, sae, keep


def _pool_sae(top_acts, top_indices, mask, num_latents):
    """(batch,seq,k) sparse -> (last_dense, mean_dense) (batch, L) en fp32."""
    import torch
    b, s, k = top_acts.shape
    dev = top_acts.device
    ta = top_acts.float()
    m = mask.float()
    lengths = m.sum(1).clamp(min=1).long() - 1
    bidx = torch.arange(b, device=dev)
    last_acts = ta[bidx, lengths, :]
    last_idx = top_indices[bidx, lengths, :].long()
    last = torch.zeros(b, num_latents, device=dev, dtype=torch.float32)
    last.scatter_add_(1, last_idx, last_acts)
    mean = torch.zeros(b, num_latents, device=dev, dtype=torch.float32)
    masked = ta * m.unsqueeze(-1)
    bids = torch.arange(b, device=dev).view(-1, 1, 1).expand(-1, s, k).reshape(-1)
    mean.index_put_((bids, top_indices.reshape(-1).long()), masked.reshape(-1), accumulate=True)
    mean = mean / m.sum(1).clamp(min=1).unsqueeze(1)
    return last, mean


def _ejecutar_worker(model_key, cache_dir, shard, nshards, batch_size, num_tok_workers):
    import torch
    from torch.utils.data import DataLoader, Dataset

    t0 = time.time()
    ds = pd.read_parquet(os.path.join(cache_dir, F_DATASET))
    authors_all = np.load(os.path.join(cache_dir, F_AUTHORS), allow_pickle=True)
    aut2idx = {str(a): i for i, a in enumerate(authors_all)}
    nu = len(authors_all)
    textos = ds["text"].astype(str).tolist()
    uidx_all = np.fromiter((aut2idx[str(a)] for a in ds["author"]), dtype=np.int64, count=len(ds))
    n = len(ds)

    # shard estriado tras ordenar por longitud (balanceado; sin carreras: npz propio)
    char_lens = np.fromiter((len(t) for t in textos), dtype=np.int64, count=n)
    order = np.argsort(char_lens, kind="stable")
    mias = order[shard::nshards]
    mias = mias[np.argsort(char_lens[mias], kind="stable")]
    print(f"[w{shard}/{nshards}] {len(mias):,}/{n:,} filas, {nu:,} usuarios", flush=True)

    device = "cuda:0"
    tok, model, run, captured, sae, keep = _setup(model_key, device)
    L = sae.cfg.num_latents
    print(f"[w{shard}/{nshards}] {model_key} cargado ({keep} capas, L={L}) {time.time()-t0:.1f}s", flush=True)

    class _DS(Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, k):
            r = int(self.rows[k]); return textos[r], r

    def _collate(batch):
        txts = [b[0] for b in batch]; rows = [b[1] for b in batch]
        t = tok(txts, max_length=CONTEXT_LEN, truncation=True, padding=True,
                return_attention_mask=True, return_tensors="pt")
        return t["input_ids"], t["attention_mask"], np.asarray(rows, dtype=np.int64)

    loader = DataLoader(_DS(mias), batch_size=batch_size, num_workers=num_tok_workers,
                        collate_fn=_collate, pin_memory=True, shuffle=False)

    H = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
    dev = torch.device(device)
    sum_pl = torch.zeros(nu, H, device=dev); sum_pm = torch.zeros(nu, H, device=dev)
    sum_sl = torch.zeros(nu, L, device=dev); sum_sm = torch.zeros(nu, L, device=dev)
    count = torch.zeros(nu, device=dev)

    total = len(mias); done = 0; last_print = time.time()
    with torch.inference_mode():
        for input_ids, attn, rows in loader:
            input_ids = input_ids.to(dev, non_blocking=True)
            attn = attn.to(dev, non_blocking=True)
            run(input_ids, attn)
            acts = captured["a"]
            mask = attn.float()
            lengths = mask.sum(1).clamp(min=1).long() - 1
            bidx = torch.arange(acts.shape[0], device=dev)
            last_tok = acts[bidx, lengths, :].float()
            mean_tok = (acts.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            top_acts, top_idx = sae.encode(acts.to(sae.dtype))
            last_sae, mean_sae = _pool_sae(top_acts, top_idx, attn, L)
            uidx = torch.as_tensor(uidx_all[rows], device=dev)
            sum_pl.index_add_(0, uidx, torch.nan_to_num(last_tok))
            sum_pm.index_add_(0, uidx, torch.nan_to_num(mean_tok))
            sum_sl.index_add_(0, uidx, torch.nan_to_num(last_sae))
            sum_sm.index_add_(0, uidx, torch.nan_to_num(mean_sae))
            count.index_add_(0, uidx, torch.ones(len(rows), device=dev))
            done += len(rows)
            now = time.time()
            if now - last_print >= 60 or done >= total:
                cps = done / max(1e-9, now - t0)
                print(f"[w{shard}/{nshards}] {100*done/total:5.1f}% ({done:,}/{total:,}) "
                      f"{cps:.0f} c/s ETA {(total-done)/max(1e-9,cps)/3600:.1f}h", flush=True)
                last_print = now

    np.savez(os.path.join(cache_dir, F_SHARD.format(i=shard, n=nshards)),
             sum_pure_last=sum_pl.cpu().numpy(), sum_pure_mean=sum_pm.cpu().numpy(),
             sum_sae_last=sum_sl.cpu().numpy(), sum_sae_mean=sum_sm.cpu().numpy(),
             count=count.cpu().numpy())
    print(f"[w{shard}/{nshards}] HECHO en {(time.time()-t0)/3600:.2f}h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model", required=True, choices=list(MODELOS))
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-tok-workers", type=int, default=8)
    args = ap.parse_args()
    if not args.worker:
        raise SystemExit("Usar como modulo (cargar_o_extraer_userlevel) o --worker.")
    _ejecutar_worker(args.model, args.cache_dir, args.shard, args.nshards,
                     args.batch_size, args.num_tok_workers)


if __name__ == "__main__":
    main()
