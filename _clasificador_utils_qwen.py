"""Ayuda para clasificadores con activaciones de Qwen.
"""
from __future__ import annotations
def truncar_modelo_para_hookpoint(model, hookpoint: str):
    import torch
    parts = hookpoint.split(".")
    if len(parts) < 3:
        raise ValueError(
            f"Hookpoint inesperado '{hookpoint}'. Formato esperado: '<trunk_path>.<list_attr>.<idx>'")
    idx = int(parts[-1])
    list_attr = parts[-2]
    trunk_path = ".".join(parts[:-2])
    trunk = model.get_submodule(trunk_path)
    layers = getattr(trunk, list_attr)
    keep = idx + 1
    if keep > len(layers):
        raise ValueError(
            f"idx={idx} fuera de rango: el modelo solo tiene {len(layers)} capas en '{trunk_path}.{list_attr}'")
    setattr(trunk, list_attr, torch.nn.ModuleList(layers[:keep]))
    return trunk, trunk_path, keep
