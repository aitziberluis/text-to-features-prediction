"""Utilidades comunes a los clasificadores (genero / edad / mbti)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple


def selection_score(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    """Tupla usada para seleccionar el mejor modelo: (f1_macro, recall_macro, precision_macro)."""
    return (
        float(metrics.get("f1_macro", float("-inf"))),
        float(metrics.get("recall_macro", float("-inf"))),
        float(metrics.get("precision_macro", float("-inf"))),
    )


def select_best_per_level(
    all_results: Dict[str, Dict[str, float]],
) -> Dict[str, Optional[Tuple[str, Dict[str, float]]]]:
    """Devuelve el mejor run por nivel ("comentario" y "usuario").

    Criterio de seleccion (de mas a menos importante): f1_macro, recall_macro,
    precision_macro y, en caso de empate, nombre alfabetico.
    """
    best: Dict[str, Optional[Tuple[str, Dict[str, float]]]] = {
        "comentario": None,
        "usuario": None,
    }
    for run_name, metrics in all_results.items():
        if run_name.startswith("test_"):
            continue
        if run_name.startswith("comentario_"):
            level = "comentario"
        elif run_name.startswith("usuario_"):
            level = "usuario"
        else:
            continue
        score = (selection_score(metrics), run_name)
        current = best[level]
        if current is None:
            best[level] = (run_name, metrics)
        else:
            current_score = (selection_score(current[1]), current[0])
            if score > current_score:
                best[level] = (run_name, metrics)
    return best


def print_best_per_level_eval(
    best_per_level: Dict[str, Optional[Tuple[str, Dict[str, float]]]],
) -> None:
    print("\n" + "=" * 70, flush=True)
    print("MEJOR MODELO POR NIVEL EN EVAL", flush=True)
    for level in ("comentario", "usuario"):
        entry = best_per_level.get(level)
        if entry is None:
            print(f"  [{level}] (no hay runs)")
            continue
        name, m = entry
        print(
            f"  [{level}] {name} | "
            f"F1 macro={m.get('f1_macro', float('nan')):.4f} | "
            f"Recall macro={m.get('recall_macro', float('nan')):.4f} | "
            f"Precision macro={m.get('precision_macro', float('nan')):.4f}",
            flush=True,
        )


def print_best_per_level_test(
    best_per_level_test: Dict[str, Optional[Tuple[str, Dict[str, float]]]],
) -> None:
    print("\n" + "=" * 70, flush=True)
    print("MEJOR MODELO POR NIVEL EVALUADO EN TEST", flush=True)
    for level in ("comentario", "usuario"):
        entry = best_per_level_test.get(level)
        if entry is None:
            print(f"  [{level}] (sin evaluacion en test)")
            continue
        name, m = entry
        print(
            f"  [{level}] {name} | "
            f"Accuracy={m.get('accuracy', float('nan')):.4f} | "
            f"BalAcc={m.get('balanced_accuracy', float('nan')):.4f} | "
            f"F1 macro={m.get('f1_macro', float('nan')):.4f} | "
            f"Recall macro={m.get('recall_macro', float('nan')):.4f} | "
            f"Precision macro={m.get('precision_macro', float('nan')):.4f}",
            flush=True,
        )
