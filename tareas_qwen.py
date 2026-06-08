"""Capa de tareas para el cache de activaciones de Qwen.

Define, por experimento (genero / edad / 4 rasgos MBTI):
  * donde vive su cache de activaciones (los 4 rasgos MBTI COMPARTEN uno solo),
  * como construir el dataframe "canonico" que se extrae,
  * `obtener_vistas(task, df_tarea)` -> (last_token, mean_token, hidden_size)
    como VistaActivaciones alineadas al orden de `df_tarea`.

Los 4 rasgos MBTI son ~99.9% los mismos comentarios, asi que se extrae UN
canonico (comentarios de autores con CUALQUIER dimension MBTI conocida) y cada
rasgo mapea sus comentarios al cache via `comment_id`.

Config por entorno:
  QWEN_GPUS="0,1,2,3,4,5"   GPUs a usar (si no, autodetecta libres)
  QWEN_BATCH="256"          batch size de extraccion
  QWEN_CACHE_BASE="/hdd/aitziber.l/cache_qwen_raw"
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import pandas as pd

from extraer_qwen import (
    CacheActivaciones, VistaActivaciones, F_META,
    cargar_o_extraer_cache, detectar_gpus_libres,
)
from preprocesamiento import (
    cargar_comentarios, cargar_autores,
    preparar_dataset_para_sae, preparar_dataset_para_edad,
)

PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"
MBTI_COLS = ["introverted", "intuitive", "thinking", "perceiving"]

CACHE_BASE = os.environ.get("QWEN_CACHE_BASE", "/hdd/aitziber.l/cache_qwen_raw")

# tareas validas: genero, edad, + cada rasgo MBTI
TAREAS = ["genero", "edad"] + MBTI_COLS


def gpus_configuradas() -> Optional[List[int]]:
    env = os.environ.get("QWEN_GPUS", "").strip()
    if env:
        return [int(x) for x in env.replace(" ", "").split(",") if x != ""]
    return None  # autodetectar


def batch_configurado() -> int:
    return int(os.environ.get("QWEN_BATCH", "256"))


def cache_dir_para(task: str) -> str:
    """genero/edad tienen su propio cache; los 4 rasgos MBTI comparten 'mbti'."""
    if task in MBTI_COLS:
        return os.path.join(CACHE_BASE, "mbti")
    return os.path.join(CACHE_BASE, task)


def _cache_listo(cache_dir: str) -> bool:
    return os.path.exists(os.path.join(cache_dir, F_META))


def _asegurar_text(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns and "body" in df.columns:
        df = df.rename(columns={"body": "text"})
    return df


def _max_comments() -> Optional[int]:
    v = os.environ.get("QWEN_MAX_COMMENTS", "").strip()
    return int(v) if v else None


def construir_df_mbti_canonico() -> pd.DataFrame:
    """Comentarios de autores con CUALQUIER dimension MBTI conocida (union).

    Devuelve df con columnas: comment_id, text, author (+ las 4 dims por si se
    quieren reutilizar). El orden define el layout fisico del cache compartido.
    Respeta QWEN_MAX_COMMENTS para smoke-tests.
    """
    df_com = cargar_comentarios(PATH_COMENTARIOS, nrows=_max_comments())  # incluye comment_id
    df_aut = cargar_autores(PATH_AUTORES)
    presentes = [c for c in MBTI_COLS if c in df_aut.columns]
    if not presentes:
        raise ValueError(f"Ninguna columna MBTI {MBTI_COLS} en {PATH_AUTORES}")
    mask = df_aut[presentes].notna().any(axis=1)
    aut = df_aut.loc[mask, ["author"] + presentes].copy()
    df = df_com.merge(aut, on="author", how="inner")
    df = _asegurar_text(df)
    df = df.dropna(subset=["text"]).reset_index(drop=True)  # comment_id se conserva
    print(f"[tareas_qwen] MBTI canonico: {len(df):,} comentarios, "
          f"{df['author'].nunique():,} autores", flush=True)
    return df


def _df_canonico(task: str, df_tarea: pd.DataFrame) -> pd.DataFrame:
    """Dataframe a extraer para la tarea (para MBTI, la union de los 4 rasgos)."""
    if task in MBTI_COLS:
        return construir_df_mbti_canonico()
    return _asegurar_text(df_tarea)


# ---- unidades de cache (lo que extrae el orquestador) ----------------------
# genero y edad tienen su cache; los 4 rasgos MBTI comparten 'mbti'.
EXTRACT_UNITS = ["genero", "edad", "mbti"]


def cache_dir_unidad(unit: str) -> str:
    return os.path.join(CACHE_BASE, unit)


def construir_df_extraccion(unit: str) -> pd.DataFrame:
    """Construye el dataframe canonico (text + comment_id [+author]) a cachear.

    Respeta QWEN_MAX_COMMENTS (smoke-test) para genero/edad; MBTI lo aplica via
    construir_df_mbti_canonico().
    """
    mc = _max_comments()
    if unit == "genero":
        df, _ = preparar_dataset_para_sae(
            path_comentarios=PATH_COMENTARIOS, path_autores=PATH_AUTORES,
            max_comments=mc, solo_genero_conocido=True)
        df = _asegurar_text(df)
        return df.dropna(subset=["text"]).reset_index(drop=True)
    if unit == "edad":
        df, _ = preparar_dataset_para_edad(
            path_comentarios=PATH_COMENTARIOS, path_autores=PATH_AUTORES,
            max_comments=mc)
        df = _asegurar_text(df)
        return df.dropna(subset=["text"]).reset_index(drop=True)
    if unit == "mbti":
        return construir_df_mbti_canonico()
    raise ValueError(f"Unidad de extraccion desconocida '{unit}'. Validas: {EXTRACT_UNITS}")


def obtener_vistas(
    task: str,
    df_tarea: pd.DataFrame,
    gpus: Optional[List[int]] = None,
) -> Tuple[VistaActivaciones, VistaActivaciones, int]:
    """Devuelve (last_token, mean_token, hidden_size) alineados a df_tarea.

    df_tarea debe tener columnas 'text'/'body' y 'comment_id'. Extrae el cache
    (canonico) si hace falta; en reejecuciones, carga y mapea al instante.
    """
    if task not in TAREAS:
        raise ValueError(f"Tarea desconocida '{task}'. Validas: {TAREAS}")
    df_tarea = _asegurar_text(df_tarea)
    if "comment_id" not in df_tarea.columns:
        raise ValueError("df_tarea no tiene 'comment_id' (usa cargar_comentarios actualizado).")
    cache_dir = cache_dir_para(task)
    ids = df_tarea["comment_id"].to_numpy()

    # camino rapido: cache existente que cubre la tarea
    if _cache_listo(cache_dir):
        cache = CacheActivaciones(cache_dir)
        try:
            pos = cache.indices_para(ids)
            print(f"[tareas_qwen] '{task}': cache listo en {cache_dir}, "
                  f"mapeando {len(ids):,} comentarios.", flush=True)
            return (VistaActivaciones(cache.last_token, pos),
                    VistaActivaciones(cache.mean_token, pos),
                    cache.hidden_size)
        except KeyError:
            print(f"[tareas_qwen] '{task}': el cache no cubre la tarea; re-extrayendo.", flush=True)

    # extraer
    if gpus is None:
        gpus = gpus_configuradas() or detectar_gpus_libres()
    df_canon = _df_canonico(task, df_tarea)
    cache = cargar_o_extraer_cache(cache_dir, df_canon, gpus=gpus, batch_size=batch_configurado())
    pos = cache.indices_para(ids)
    return (VistaActivaciones(cache.last_token, pos),
            VistaActivaciones(cache.mean_token, pos),
            cache.hidden_size)
