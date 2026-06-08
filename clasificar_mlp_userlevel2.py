"""Clasificador MLP a nivel de USUARIO sobre features de GPT-2 / Qwen.
Para cada (modelo in {gpt2, qwen}, tarea in {genero, edad, introverted,
intuitive, thinking, perceiving}) entrena un MLP por cada combinacion de:
    representacion in {pure, sae}   x   pooling in {last, mean}
es decir 4 MLPs por (modelo, tarea). Arquitectura identica para pura y SAE
(solo cambia la dimension de entrada), para una comparacion justa.
- Features user-level desde el cache slim (extraer_userlevel).
- Etiquetas por usuario desde author_profiles.
- Split de usuarios 70/15/15 estratificado (seed 42), cacheado por tarea y
  reutilizado en todas las representaciones/poolings/modelos de esa tarea.
- StandardScaler (fit en train), MLP (256->128, LayerNorm+GELU, Dropout 0.4),
  AdamW, perdida con pesos de clase balanceados, early-stopping en F1-macro de
  eval. Metricas finales en test.
Resultados -> modelos/<tarea>_<modelo>_mlp_userlevel/resultados_resumen.json
(no toca nada existente).
CLI:
    python clasificar_mlp_userlevel.py --model qwen --task genero
    python clasificar_mlp_userlevel.py --all          # 2 modelos x 6 tareas
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extraer_userlevel import cargar_o_extraer_userlevel, cargar_userlevel_si_existe
from tareas_qwen import construir_df_extraccion, gpus_configuradas, batch_configurado
from preprocesamiento import (cargar_autores, normalizar_genero, normalizar_edad,
                              DEFAULT_AGE_BINS, DEFAULT_AGE_GROUP_LABELS)

PATH_AUTORES = "data/author_profiles.csv"
MBTI_COLS = ["introverted", "intuitive", "thinking", "perceiving"]
TASKS = ["genero", "edad"] + MBTI_COLS
MODELS = ["gpt2", "qwen"]
SEED = 42
# configurables por entorno (defaults segun layout del proyecto, propietario aitziber.l)
SPLITS_BASE = os.environ.get("MLP_SPLITS", "data/splits_mlp")   # nuevo, no pisa los existentes
OUT_BASE = os.environ.get("MLP_OUT", "modelos")                 # subdirs nuevos *_mlp_userlevel

# hiperparametros MLP
HIDDEN = (256, 128)
DROPOUT = 0.4
LR = 1e-3
WEIGHT_DECAY = 1e-2
MAX_EPOCHS = 300
PATIENCE = 30
MLP_BATCH = 256


def unit_de_tarea(task: str) -> str:
    return "mbti" if task in MBTI_COLS else task


def _perfil_autores() -> pd.DataFrame:
    df = cargar_autores(PATH_AUTORES)
    df = normalizar_genero(df)
    df = normalizar_edad(df, age_bins=DEFAULT_AGE_BINS, age_labels=DEFAULT_AGE_GROUP_LABELS)
    return df


def etiquetas_por_usuario(task: str, authors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Devuelve (labels, mask_validos, class_names) alineados a `authors`.

    mask_validos indica que usuarios tienen etiqueta conocida para la tarea.
    """
    perfil = _perfil_autores().drop_duplicates("author").set_index("author")
    if task == "genero":
        col = perfil["gender_clean"].reindex(authors)
        mapping = {"f": 0, "m": 1}
        labels = np.array([mapping.get(v, -1) for v in col], dtype=np.int64)
        class_names = ["f", "m"]
    elif task == "edad":
        col = perfil["age_group"].reindex(authors)
        mapping = {g: i for i, g in enumerate(DEFAULT_AGE_GROUP_LABELS)}
        labels = np.array([mapping.get(v, -1) for v in col], dtype=np.int64)
        class_names = list(DEFAULT_AGE_GROUP_LABELS)
    elif task in MBTI_COLS:
        col = perfil[task].reindex(authors)
        labels = np.array([int(v) if pd.notna(v) else -1 for v in col], dtype=np.int64)
        class_names = ["0", "1"]
    else:
        raise ValueError(f"tarea desconocida {task}")
    mask = labels >= 0
    return labels, mask, class_names


def split_usuarios(task: str, authors: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    """Split 70/15/15 estratificado por usuario, cacheado por tarea."""
    d = os.path.join(SPLITS_BASE, task)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "split_usuarios.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        cached = set(z["train"]) | set(z["eval"]) | set(z["test"])
        if cached == set(authors.tolist()):
            return {"train": z["train"], "eval": z["eval"], "test": z["test"]}
        print(f"[mlp] split cacheado de {task} no coincide; regenerando", flush=True)
    tr_ev, te = train_test_split(authors, test_size=0.15, random_state=SEED, stratify=labels)
    lab_tr_ev = labels[np.isin(authors, tr_ev)]
    tr, ev = train_test_split(tr_ev, test_size=0.15 / 0.85, random_state=SEED, stratify=lab_tr_ev)
    np.savez(p, train=tr, eval=ev, test=te)
    return {"train": tr, "eval": ev, "test": te}


# ---- MLP -------------------------------------------------------------------
def _build_mlp(in_dim: int, n_classes: int):
    import torch.nn as nn
    layers = []
    prev = in_dim
    for h in HIDDEN:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(DROPOUT)]
        prev = h
    layers += [nn.Linear(prev, n_classes)]
    return nn.Sequential(*layers)


def _evaluar(y_true, y_pred, class_names) -> Dict[str, object]:
    labels_idx = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_per_class": [float(x) for x in f1_score(y_true, y_pred, average=None, labels=labels_idx, zero_division=0)],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_idx).tolist(),
    }


def entrenar_mlp(Xtr, ytr, Xev, yev, Xte, yte, n_classes, class_names, device) -> Tuple[Dict, Dict]:
    import torch
    import torch.nn as nn
    torch.manual_seed(SEED); np.random.seed(SEED)
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).astype(np.float32)
    Xev = sc.transform(Xev).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)

    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr)
    w = torch.tensor(cw, dtype=torch.float32, device=device)
    model = _build_mlp(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss(weight=w)

    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
    Xev_t = torch.tensor(Xev, device=device)
    Xte_t = torch.tensor(Xte, device=device)
    n = len(ytr_t)

    best_f1, best_state, best_epoch, since = -1.0, None, -1, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, MLP_BATCH):
            idx = perm[s:s + MLP_BATCH]
            opt.zero_grad()
            loss = crit(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            yev_pred = model(Xev_t).argmax(1).cpu().numpy()
        f1 = f1_score(yev, yev_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_epoch, since = f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        yev_pred = model(Xev_t).argmax(1).cpu().numpy()
        yte_pred = model(Xte_t).argmax(1).cpu().numpy()
    eval_m = _evaluar(yev, yev_pred, class_names)
    test_m = _evaluar(yte, yte_pred, class_names)
    eval_m["best_epoch"] = int(best_epoch)
    return eval_m, test_m


def run(model_key: str, task: str, gpus: Optional[List[int]] = None) -> Dict:
    import torch
    print(f"\n\nMLP user-level | modelo={model_key} | tarea={task}\n", flush=True)
    unit = unit_de_tarea(task)
    cache = cargar_userlevel_si_existe(model_key, unit)
    if cache is None:
        df_canon = construir_df_extraccion(unit)   # recarga CSV solo si hay que extraer
        cache = cargar_o_extraer_userlevel(model_key, unit, df_canon, gpus=gpus,
                                           batch_size=batch_configurado())
    authors = np.asarray([str(a) for a in cache.authors])
    labels_all, mask, class_names = etiquetas_por_usuario(task, authors)
    authors_v = authors[mask]
    labels_v = labels_all[mask]
    print(f"usuarios con etiqueta: {len(authors_v):,}/{len(authors):,} | "
          f"dist={np.bincount(labels_v).tolist()} | clases={class_names}", flush=True)

    sp = split_usuarios(task, authors_v, labels_v)
    a2i = {a: i for i, a in enumerate(authors)}              # indice en el cache
    av2i = {a: i for i, a in enumerate(authors_v)}           # indice en el subset etiquetado
    def rows(auths, base):
        return np.array([base[a] for a in auths], dtype=np.int64)
    idx_cache = {k: rows(sp[k], a2i) for k in ("train", "eval", "test")}
    idx_lab = {k: rows(sp[k], av2i) for k in ("train", "eval", "test")}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resultados = {}
    for rep in ("pure", "sae"):
        for pool in ("last", "mean"):
            feats = cache.features(rep, pool)
            Xtr = feats[idx_cache["train"]]; ytr = labels_v[idx_lab["train"]]
            Xev = feats[idx_cache["eval"]];  yev = labels_v[idx_lab["eval"]]
            Xte = feats[idx_cache["test"]];  yte = labels_v[idx_lab["test"]]
            t0 = time.time()
            eval_m, test_m = entrenar_mlp(Xtr, ytr, Xev, yev, Xte, yte,
                                          len(class_names), class_names, device)
            run_name = f"usuario_{rep}_{pool}"
            resultados[run_name] = {"eval_metrics": eval_m, "test_metrics": test_m,
                                    "in_dim": int(Xtr.shape[1])}
            print(f"{run_name:22s} in={Xtr.shape[1]:5d} | "
                  f"eval Acc={eval_m['accuracy']:.4f} Prec={eval_m['precision_macro']:.4f} "
                  f"Recall={eval_m['recall_macro']:.4f} F1={eval_m['f1_macro']:.4f} "
                  f"(ep{eval_m['best_epoch']}) | "
                  f"test Acc={test_m['accuracy']:.4f} Prec={test_m['precision_macro']:.4f} "
                  f"Recall={test_m['recall_macro']:.4f} F1={test_m['f1_macro']:.4f} "
                  f"BalAcc={test_m['balanced_accuracy']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    best = max(resultados.items(), key=lambda kv: kv[1]["eval_metrics"]["f1_macro"])
    out_dir = os.path.join(OUT_BASE, f"{task}_{model_key}_mlp_userlevel")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "model": MODELS_INFO(model_key), "task": task, "level": "usuario",
        "arch": {"hidden": list(HIDDEN), "dropout": DROPOUT, "lr": LR,
                 "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE},
        "class_names": class_names,
        "n_users_labeled": int(len(authors_v)),
        "split_sizes": {k: int(len(sp[k])) for k in sp},
        "runs": resultados,
        "best_run_on_eval": {"name": best[0], **best[1]},
    }
    with open(os.path.join(out_dir, "resultados_resumen.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"-> guardado {out_dir}/resultados_resumen.json | mejor={best[0]} "
          f"(test Acc={best[1]['test_metrics']['accuracy']:.4f} "
          f"Prec={best[1]['test_metrics']['precision_macro']:.4f} "
          f"Recall={best[1]['test_metrics']['recall_macro']:.4f} "
          f"F1={best[1]['test_metrics']['f1_macro']:.4f})", flush=True)
    return summary


def MODELS_INFO(model_key: str) -> str:
    from extraer_userlevel import MODELOS
    return MODELOS[model_key]["model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gpus", default=None)
    args = ap.parse_args()
    gpus = ([int(x) for x in args.gpus.replace(" ", "").split(",") if x]
            if args.gpus else (gpus_configuradas()))
    if args.all:
        for m in MODELS:
            for t in TASKS:
                run(m, t, gpus=gpus)
    else:
        if not (args.model and args.task):
            raise SystemExit("Usa --all o (--model y --task)")
        run(args.model, args.task, gpus=gpus)


if __name__ == "__main__":
    main()
