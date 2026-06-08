"""Phase 1 — Layer-probe sweep for picking the best layer of a causal LM.

Idea:
  - One forward pass over a sample of comments
  - Hook EVERY transformer layer; mean-pool the hidden state on GPU per layer
  - For each (layer, task) pair train a fast LogReg probe; report eval F1-macro

Tasks probed:
  - genero       (binary: m/f, from gender_clean)
  - edad         (4-class: 14_19, 20_29, 30_39, 40_plus)
  - mbti_intro   (binary: introverted)
  - mbti_intu    (binary: intuitive)
  - mbti_think   (binary: thinking)
  - mbti_perc    (binary: perceiving)

Outputs:
  - data/layer_sweep_<model_tag>/activations_layer_<L>.npy        (fp16, (n, hidden))
  - data/layer_sweep_<model_tag>/labels.parquet                   (n rows, all task columns)
  - data/layer_sweep_<model_tag>/probe_results.csv                (layer, task, metric)

Defaults to Qwen3-4B-Base; override via SAE_MODEL env var.
"""
import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocesamiento import cargar_autores, cargar_comentarios

# CONFIG -----------------------------------------------------------------

MODEL = os.getenv("SAE_MODEL", "Qwen/Qwen3-4B-Base")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CONTEXT_LEN = int(os.getenv("LAYER_SWEEP_CONTEXT_LEN", "256"))
SAMPLE_N = int(os.getenv("LAYER_SWEEP_SAMPLE", "100000"))
EXTRACT_BATCH_SIZE = int(os.getenv("LAYER_SWEEP_BATCH", "32"))
NUM_WORKERS = int(os.getenv("LAYER_SWEEP_WORKERS", "16"))
EVERY_NTH = int(os.getenv("LAYER_SWEEP_EVERY", "1"))  # 1 = every layer
SEED = 42

PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"
TEXT_COLUMN = "body"

AGE_GROUPS = ["14_19", "20_29", "30_39", "40_plus"]


# DATA -------------------------------------------------------------------

def _age_to_group(age) -> str:
    try:
        a = int(age)
    except (TypeError, ValueError):
        return ""
    if 14 <= a <= 19:
        return "14_19"
    if 20 <= a <= 29:
        return "20_29"
    if 30 <= a <= 39:
        return "30_39"
    if a >= 40:
        return "40_plus"
    return ""


def cargar_dataset_multitarea() -> pd.DataFrame:
    """Carga comentarios + perfil de autor con TODAS las etiquetas relevantes."""
    print("Cargando comentarios...")
    df_c = cargar_comentarios(PATH_COMENTARIOS, nrows=None)
    print(f"  comentarios: {len(df_c):,}")
    print("Cargando autores...")
    df_a = cargar_autores(PATH_AUTORES)
    print(f"  autores: {len(df_a):,}")

    if "gender" in df_a.columns:
        df_a["gender"] = df_a["gender"].astype(str).str.lower().str.strip()
    df_a["gender_clean"] = pd.NA
    if "is_female" in df_a.columns:
        df_a.loc[df_a["is_female"] == 1, "gender_clean"] = "f"
        df_a.loc[df_a["is_female"] == 0, "gender_clean"] = "m"
    if "gender" in df_a.columns:
        df_a.loc[df_a["gender_clean"].isna() & (df_a["gender"] == "f"), "gender_clean"] = "f"
        df_a.loc[df_a["gender_clean"].isna() & (df_a["gender"] == "m"), "gender_clean"] = "m"

    df_a["age_group"] = df_a["age"].apply(_age_to_group)

    keep_cols = [
        "author", "gender_clean", "age_group",
        "introverted", "intuitive", "thinking", "perceiving",
    ]
    df_a = df_a[[c for c in keep_cols if c in df_a.columns]].copy()

    print("Uniendo comentarios con etiquetas por author...")
    df = df_c.merge(df_a, on="author", how="inner")
    if TEXT_COLUMN != "text":
        df = df.rename(columns={TEXT_COLUMN: "text"})
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    print(f"  comentarios con autor: {len(df):,}")
    return df


def muestrear_estratificado(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if len(df) <= n:
        return df.reset_index(drop=True)
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[np.sort(idx)].reset_index(drop=True)


# MODEL UTILS ------------------------------------------------------------

def _layer_list(model) -> nn.ModuleList:
    """Devuelve la lista de bloques transformer (HF: .transformer.h o .model.layers)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError(f"Arquitectura no soportada: {type(model)}")


def _hidden_size(model) -> int:
    cfg = model.config
    return getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")


# TOKEN BATCHING ---------------------------------------------------------

class _TextDataset(Dataset):
    def __init__(self, textos: List[str], order: np.ndarray):
        self.textos = textos
        self.order = order

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, i: int):
        idx = int(self.order[i])
        return self.textos[idx], idx


class _Collate:
    def __init__(self, tokenizer, context_len: int):
        self.tokenizer = tokenizer
        self.context_len = context_len

    def __call__(self, batch):
        textos = [b[0] for b in batch]
        indices = torch.tensor([b[1] for b in batch], dtype=torch.long)
        tokens = self.tokenizer(
            textos,
            max_length=self.context_len,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return tokens["input_ids"], tokens["attention_mask"], indices


# EXTRACTION -------------------------------------------------------------

def extraer_activaciones_por_capa(
    df: pd.DataFrame, sample_dir: Path
) -> Tuple[List[int], int]:
    """Forward pass una vez por todas las muestras; mean-pool por capa en GPU.

    Devuelve la lista de capas y el hidden_size.
    """
    print(f"\n[Extract] Cargando {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_fp16 = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": DEVICE},
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # Liberar lm_head (y el norm final si Qwen) — no se usa.
    if hasattr(model, "lm_head"):
        del model.lm_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layers = _layer_list(model)
    n_layers = len(layers)
    layer_indices = list(range(0, n_layers, EVERY_NTH))
    if layer_indices[-1] != n_layers - 1:
        layer_indices.append(n_layers - 1)
    print(f"[Extract] Modelo con {n_layers} capas. Sondeando capas: {layer_indices}")
    hidden_size = _hidden_size(model)

    n = len(df)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Buffers en RAM (bajo: ~100k * 36 * 2560 * 2B fp16 = 18 GB para Qwen3-4B).
    bufs = {
        L: np.empty((n, hidden_size), dtype=np.float16)
        for L in layer_indices
    }

    # Hooks por capa: mean-pool sobre tokens reales, en GPU.
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    attn_holder: Dict[str, torch.Tensor] = {}

    def make_hook(L: int):
        def _hook(module, inputs, outputs):
            out = outputs[0] if isinstance(outputs, tuple) else outputs
            captured[L] = out
        return _hook

    for L in layer_indices:
        handles.append(layers[L].register_forward_hook(make_hook(L)))

    textos = df["text"].astype(str).tolist()
    print("[Extract] Calculando longitudes (bucketing por len(text))...")
    char_lens = np.fromiter((len(t) for t in textos), dtype=np.int64, count=n)
    sorted_order = np.argsort(char_lens, kind="stable")

    dataset = _TextDataset(textos, sorted_order)
    collate = _Collate(tokenizer, CONTEXT_LEN)
    loader = DataLoader(
        dataset,
        batch_size=EXTRACT_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate,
        pin_memory=True,
        shuffle=False,
        persistent_workers=False,
    )

    total_steps = math.ceil(n / EXTRACT_BATCH_SIZE)
    device = torch.device(DEVICE)
    last_print = time.time()

    try:
        with torch.inference_mode():
            for step, (input_ids, attn_mask, original_idx) in enumerate(loader):
                input_ids = input_ids.to(device, non_blocking=True)
                attn_mask = attn_mask.to(device, non_blocking=True)
                # Llamamos al trunk (model.model) en lugar de model(...) para
                # evitar la lm_head, que hemos borrado para liberar VRAM.
                trunk = model.model if hasattr(model, "model") else model
                trunk(input_ids=input_ids, attention_mask=attn_mask)

                mask = attn_mask.to(torch.float16).unsqueeze(-1)
                denom = attn_mask.to(torch.float16).sum(dim=1, keepdim=True).clamp(min=1)
                idx_np = original_idx.numpy()

                for L in layer_indices:
                    h = captured[L]
                    pooled = (h * mask).sum(dim=1) / denom  # (B, hidden)
                    bufs[L][idx_np] = pooled.cpu().to(torch.float16).numpy()
                captured.clear()

                now = time.time()
                if now - last_print >= 60 or step == 0 or step == total_steps - 1:
                    pct = 100.0 * (step + 1) / total_steps
                    print(
                        f"  [{pct:5.1f}%] step {step+1}/{total_steps} "
                        f"({(step+1)*EXTRACT_BATCH_SIZE:,}/{n:,}, "
                        f"seq_len={input_ids.shape[1]})",
                        flush=True,
                    )
                    last_print = now
    finally:
        for h in handles:
            h.remove()

    print("[Extract] Volcando buffers a disco...")
    for L in layer_indices:
        np.save(sample_dir / f"activations_layer_{L}.npy", bufs[L])

    df.to_parquet(sample_dir / "labels.parquet", index=False)
    with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL,
                "context_len": CONTEXT_LEN,
                "n": n,
                "hidden_size": hidden_size,
                "n_layers": n_layers,
                "layer_indices": layer_indices,
                "sample_n": SAMPLE_N,
                "extract_batch_size": EXTRACT_BATCH_SIZE,
            },
            f,
            indent=2,
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return layer_indices, hidden_size


# PROBES -----------------------------------------------------------------

def _build_tasks(df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Para cada tarea, devuelve (mask, y) con las filas validas y la etiqueta entera."""
    tasks: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if "gender_clean" in df.columns:
        m = df["gender_clean"].isin(["f", "m"]).to_numpy()
        y = np.where(df["gender_clean"].fillna("?").to_numpy() == "f", 0, 1)
        tasks["genero"] = (m, y)

    if "age_group" in df.columns:
        m = df["age_group"].isin(AGE_GROUPS).to_numpy()
        label_map = {g: i for i, g in enumerate(AGE_GROUPS)}
        y = np.array(
            [label_map.get(g, -1) for g in df["age_group"].fillna("").to_numpy()],
            dtype=np.int64,
        )
        tasks["edad"] = (m, y)

    for col in ("introverted", "intuitive", "thinking", "perceiving"):
        if col in df.columns:
            m = df[col].notna().to_numpy()
            y = df[col].fillna(-1).astype(int).to_numpy()
            tasks[f"mbti_{col[:5]}"] = (m, y)
    return tasks


def correr_probes(
    sample_dir: Path, layer_indices: List[int]
) -> pd.DataFrame:
    """Para cada (capa, tarea) entrena LogReg y reporta metricas."""
    df = pd.read_parquet(sample_dir / "labels.parquet")
    tasks = _build_tasks(df)
    print(f"\n[Probe] Tareas: {list(tasks.keys())}")
    for name, (mask, y) in tasks.items():
        n_valid = int(mask.sum())
        if n_valid > 0:
            ys = y[mask]
            uniq, counts = np.unique(ys, return_counts=True)
            print(f"  {name}: n={n_valid:,} | distrib={dict(zip(uniq.tolist(), counts.tolist()))}")
        else:
            print(f"  {name}: n=0 (skip)")

    rows = []
    for L in layer_indices:
        X_full = np.load(sample_dir / f"activations_layer_{L}.npy", mmap_mode="r")
        for name, (mask, y) in tasks.items():
            if mask.sum() < 200 or len(np.unique(y[mask])) < 2:
                continue
            X = np.asarray(X_full[mask], dtype=np.float32)
            yy = y[mask]
            X_tr, X_ev, y_tr, y_ev = train_test_split(
                X, yy, test_size=0.2, random_state=SEED, stratify=yy,
            )
            sc = StandardScaler(with_mean=True, with_std=True)
            X_tr = sc.fit_transform(X_tr)
            X_ev = sc.transform(X_ev)
            t0 = time.time()
            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=200,
                C=1.0,
                class_weight="balanced",
                n_jobs=-1,
            )
            clf.fit(X_tr, y_tr)
            yp = clf.predict(X_ev)
            f1m = f1_score(y_ev, yp, average="macro", zero_division=0)
            bal = balanced_accuracy_score(y_ev, yp)
            elapsed = time.time() - t0
            rows.append(
                {
                    "layer": int(L),
                    "task": name,
                    "n_train": int(len(y_tr)),
                    "n_eval": int(len(y_ev)),
                    "f1_macro": float(f1m),
                    "balanced_acc": float(bal),
                    "fit_seconds": float(elapsed),
                }
            )
            print(
                f"  layer={L:>2} task={name:<11} "
                f"f1m={f1m:.4f} balacc={bal:.4f} ({elapsed:5.1f}s, n={len(yy):,})",
                flush=True,
            )

    res = pd.DataFrame(rows)
    res.to_csv(sample_dir / "probe_results.csv", index=False)
    print(f"\n[Probe] Resultados guardados en {sample_dir / 'probe_results.csv'}")

    if not res.empty:
        print("\n[Probe] Mejor capa por tarea (F1 macro):")
        for task, sub in res.groupby("task"):
            best = sub.loc[sub["f1_macro"].idxmax()]
            print(
                f"  {task:<11} -> capa {int(best['layer']):>2} "
                f"(f1m={best['f1_macro']:.4f}, balacc={best['balanced_acc']:.4f})"
            )

        # Mejor capa promedio entre tareas (normalizando por la mejor capa de
        # cada tarea para que tareas faciles no dominen).
        pivot = res.pivot(index="layer", columns="task", values="f1_macro")
        pivot_norm = pivot.div(pivot.max(axis=0), axis=1)
        avg_norm = pivot_norm.mean(axis=1).sort_values(ascending=False)
        print("\n[Probe] Mejor capa promedio (F1 macro normalizado):")
        for layer, score in avg_norm.head(5).items():
            print(f"  capa {int(layer):>2}: score={score:.4f}")
    return res


# MAIN -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extract", action="store_true",
                        help="Saltar el forward (usar activaciones cacheadas).")
    parser.add_argument("--skip-probe", action="store_true",
                        help="Solo extraer; no correr probes.")
    args = parser.parse_args()

    model_tag = re.sub(r"[/.]", "_", MODEL).lower()
    sample_dir = Path(f"data/layer_sweep_{model_tag}")
    print(f"Output dir: {sample_dir}")
    print(f"Modelo: {MODEL} | CONTEXT_LEN={CONTEXT_LEN} | sample={SAMPLE_N:,} | every={EVERY_NTH}")

    layer_indices = None
    if not args.skip_extract:
        df_full = cargar_dataset_multitarea()
        df = muestrear_estratificado(df_full, SAMPLE_N)
        print(f"\nMuestra: {len(df):,} comentarios")
        layer_indices, _hs = extraer_activaciones_por_capa(df, sample_dir)
    else:
        with open(sample_dir / "meta.json") as f:
            meta = json.load(f)
        layer_indices = meta["layer_indices"]
        print(f"[skip-extract] usando capas cacheadas: {layer_indices}")

    if not args.skip_probe:
        correr_probes(sample_dir, layer_indices)


if __name__ == "__main__":
    main()
