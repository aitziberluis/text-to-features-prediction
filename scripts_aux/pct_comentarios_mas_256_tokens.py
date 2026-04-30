"""Calcula el porcentaje de comentarios con mas de 256 tokens (tokenizer GPT-2).

Usa el mismo CSV que el resto del pipeline (data/all_comments_since_2015.csv) y
filtra filas con 'body' nulo/vacio. Procesa en chunks para no cargar todo en RAM.
Imprime progreso cada N comentarios y un resumen final.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
MODEL = "openai-community/gpt2"
THRESHOLD = 256
CHUNKSIZE = 200_000
PROGRESS_EVERY = 1_000_000


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / PATH_COMENTARIOS
    if not csv_path.exists():
        print(f"ERROR: no existe {csv_path}", flush=True)
        sys.exit(1)

    print(f"CSV: {csv_path}", flush=True)
    print(f"Tokenizer: {MODEL}", flush=True)
    print(f"Umbral: > {THRESHOLD} tokens", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total = 0
    over = 0
    nulls = 0
    sum_len = 0
    max_len = 0
    start = time.time()
    last_print = start

    for chunk in pd.read_csv(
        csv_path,
        usecols=["body"],
        chunksize=CHUNKSIZE,
        dtype={"body": "string"},
        low_memory=False,
    ):
        before = len(chunk)
        chunk = chunk.dropna(subset=["body"])
        chunk = chunk[chunk["body"].str.strip().astype(bool)]
        nulls += before - len(chunk)
        if chunk.empty:
            continue

        texts = chunk["body"].tolist()
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        lens = [len(ids) for ids in encoded]

        total += len(lens)
        sum_len += sum(lens)
        local_max = max(lens)
        if local_max > max_len:
            max_len = local_max
        over += sum(1 for n in lens if n > THRESHOLD)

        now = time.time()
        if total - (total % PROGRESS_EVERY) > 0 and now - last_print >= 30:
            pct = 100.0 * over / max(total, 1)
            elapsed = now - start
            speed = total / max(elapsed, 1e-6)
            print(
                f"  [progreso] procesados={total:,} | >{THRESHOLD}={over:,} ({pct:.2f}%) | "
                f"avg_len={sum_len/max(total,1):.1f} | max_len={max_len} | "
                f"vel={speed:,.0f} c/s | t={elapsed/60:.1f} min",
                flush=True,
            )
            last_print = now

    elapsed = time.time() - start
    pct = 100.0 * over / max(total, 1)
    avg_len = sum_len / max(total, 1)
    print("\n" + "=" * 60, flush=True)
    print("RESUMEN", flush=True)
    print(f"Comentarios validos procesados : {total:,}", flush=True)
    print(f"Comentarios body nulo/vacio    : {nulls:,}", flush=True)
    print(f"Comentarios con > {THRESHOLD} tokens : {over:,}", flush=True)
    print(f"Porcentaje                     : {pct:.4f}%", flush=True)
    print(f"Longitud media (tokens)        : {avg_len:.2f}", flush=True)
    print(f"Longitud maxima (tokens)       : {max_len}", flush=True)
    print(f"Tiempo total                   : {elapsed/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
