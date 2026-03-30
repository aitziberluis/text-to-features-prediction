"""
Dashboard interactivo para analisis de la SAE entrenada sobre GPT-2.

Inspirado en SAEDashboard y SAELens. Funcionalidades:

1. FEATURE BROWSER: Explora las 16384 features de la SAE
   - Top tokens que activan cada feature (decoder weights)
   - Distribucion de activaciones por feature
   - Ejemplos de comentarios que activan fuertemente cada feature

2. ANALISIS DE GÉNERO: Qué features son mas discriminativas para M/F
   - Features con mayor diferencia de activacion media entre generos
   - Visualizacion de los tokens asociados a features "de genero"

3. ANALISIS DE EDAD: Features discriminativas por rango de edad
   - Features con mayor activacion diferencial por grupo de edad

4. FEATURE STATISTICS: Estadisticas globales
   - Frecuencia de activacion de cada feature (sparsity)
   - Correlaciones entre features

5. LOGIT LENS: Qué predice cada feature en el espacio de vocabulario
   - Proyeccion del decoder weight al espacio de embeddings/unembeddings

Lanza un servidor Gradio local.

Uso:
    python sae_dashboard.py
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiny_sae import Sae

# =====================
# CONFIGURACION
# =====================

MODEL = "openai-community/gpt2"
PATH_SAE = "sae-ckpts/sae-gpt2-comments"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Datos de activaciones
SAE_GENDER_DIR = "/hdd/aitziber.l/activaciones_sae_gpt2_genero"
GPT_AGE_DIR = "data/activaciones_gpt2_edad"

# Splits
SPLITS_GENDER_DIR = "data/splits_genero_70_15_15"
SPLITS_AGE_DIR = "data/splits_edad"

# Datos originales
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"

# Cuántos top tokens mostrar por feature
TOP_K_TOKENS = 20
# Cuántos ejemplos mostrar por feature
NUM_EXAMPLES = 5
# Batch para procesar texto
BATCH_SIZE = 32
CONTEXT_LEN = 512


# =====================
# CARGA GLOBAL (una sola vez)
# =====================


def load_sae():
    """Carga la SAE."""
    print("Cargando SAE...")
    sae = Sae.load_from_disk(PATH_SAE, device=DEVICE)
    print(f"  SAE: {sae.cfg.num_latents} latentes, k={sae.cfg.k}, hookpoint={sae.cfg.hookpoint}")
    return sae


def load_tokenizer_and_model():
    """Carga tokenizer y modelo GPT-2."""
    print("Cargando GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": DEVICE},
        dtype=torch.float32,
    )
    model.eval()
    return tokenizer, model


def load_gender_activations():
    """Carga activaciones SAE de genero (memmap)."""
    meta_path = os.path.join(SAE_GENDER_DIR, "meta.json")
    if not os.path.exists(meta_path):
        print(f"  No se encontraron activaciones de genero en {SAE_GENDER_DIR}")
        return None, None, None
    with open(meta_path) as f:
        meta = json.load(f)
    n, num_latents = meta["n"], meta["num_latents"]

    last_token = np.memmap(
        os.path.join(SAE_GENDER_DIR, "last_token.mmap"),
        dtype=np.float16, mode="r", shape=(n, num_latents),
    )
    labels = np.load(os.path.join(SAE_GENDER_DIR, "labels.npy"))
    print(f"  Genero: {n:,} comentarios, {num_latents} latentes")
    return last_token, labels, num_latents


# =====================
# INICIALIZACION
# =====================

print("=" * 60)
print("INICIALIZANDO SAE DASHBOARD")
print("=" * 60)

sae = load_sae()
tokenizer, gpt2_model = load_tokenizer_and_model()
gender_acts, gender_labels, gender_num_latents = load_gender_activations()

NUM_LATENTS = sae.cfg.num_latents
W_dec = sae.W_dec.data.detach()  # (num_latents, d_in=768)
b_dec = sae.b_dec.data.detach()  # (d_in,)
W_enc = sae.encoder.weight.data.detach()  # (num_latents, d_in)
b_enc = sae.encoder.bias.data.detach()  # (num_latents,)

# Obtener la unembedding matrix de GPT-2 para logit lens
W_U = gpt2_model.lm_head.weight.data.detach()  # (vocab_size, d_model)

print("Inicializacion completada.")


# =====================
# 1. FEATURE BROWSER - Top tokens por decoder weight
# =====================


def get_top_tokens_for_feature(feature_idx: int, top_k: int = TOP_K_TOKENS) -> str:
    """Para una feature dada, encuentra los top tokens en el vocabulario
    que mas se alinean con su decoder weight direction (logit lens)."""

    if feature_idx < 0 or feature_idx >= NUM_LATENTS:
        return f"Feature index debe estar entre 0 y {NUM_LATENTS - 1}"

    # Decoder weight de esta feature: (d_in,)
    dec_vec = W_dec[feature_idx]

    # Logit lens: proyectar al espacio de vocabulario
    # logits = W_U @ dec_vec  -> (vocab_size,)
    logits = W_U @ dec_vec  # (vocab_size,)

    top_vals, top_ids = logits.topk(top_k)
    bot_vals, bot_ids = logits.topk(top_k, largest=False)

    lines = [f"## Feature {feature_idx} - Top {top_k} tokens (Logit Lens)\n"]
    lines.append("### Tokens que esta feature PROMUEVE:\n")
    lines.append("| Rank | Token | Logit |")
    lines.append("|------|-------|-------|")
    for i in range(top_k):
        token = tokenizer.decode([top_ids[i].item()])
        token_repr = repr(token)
        lines.append(f"| {i+1} | {token_repr} | {top_vals[i].item():.3f} |")

    lines.append(f"\n### Tokens que esta feature SUPRIME:\n")
    lines.append("| Rank | Token | Logit |")
    lines.append("|------|-------|-------|")
    for i in range(top_k):
        token = tokenizer.decode([bot_ids[i].item()])
        token_repr = repr(token)
        lines.append(f"| {i+1} | {token_repr} | {bot_vals[i].item():.3f} |")

    return "\n".join(lines)


def get_feature_stats(feature_idx: int) -> str:
    """Estadisticas basicas del decoder weight de una feature."""
    if feature_idx < 0 or feature_idx >= NUM_LATENTS:
        return "Index fuera de rango"

    dec_vec = W_dec[feature_idx]
    enc_vec = W_enc[feature_idx]

    dec_norm = dec_vec.norm().item()
    enc_norm = enc_vec.norm().item()
    enc_bias = b_enc[feature_idx].item()
    cosine_sim = F.cosine_similarity(dec_vec.unsqueeze(0), enc_vec.unsqueeze(0)).item()

    lines = [
        f"## Feature {feature_idx} - Estadísticas\n",
        f"- **Decoder weight norm**: {dec_norm:.4f}",
        f"- **Encoder weight norm**: {enc_norm:.4f}",
        f"- **Encoder bias**: {enc_bias:.4f}",
        f"- **Cosine sim (enc ↔ dec)**: {cosine_sim:.4f}",
    ]
    return "\n".join(lines)


def plot_feature_decoder_histogram(feature_idx: int):
    """Histograma de los valores del decoder weight."""
    if feature_idx < 0 or feature_idx >= NUM_LATENTS:
        return None

    dec_vec = W_dec[feature_idx].cpu().numpy()
    fig = px.histogram(
        x=dec_vec, nbins=50,
        title=f"Feature {feature_idx} - Distribución de decoder weights",
        labels={"x": "Weight value", "y": "Count"},
    )
    fig.update_layout(height=350)
    return fig


# =====================
# 2. ANALISIS DE TEXTO - Activar SAE sobre un texto
# =====================


def analyze_text(text: str, top_k_features: int = 10) -> tuple:
    """Pasa un texto por GPT-2 + SAE y muestra las features mas activas."""
    if not text.strip():
        return "Introduce un texto.", None

    tokens = tokenizer(
        text, max_length=CONTEXT_LEN, truncation=True,
        padding=False, return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(DEVICE)

    hookpoint = gpt2_model.get_submodule(sae.cfg.hookpoint)
    captured = {}

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        captured["act"] = outputs.detach()

    handle = hookpoint.register_forward_hook(hook)
    try:
        with torch.no_grad():
            gpt2_model.transformer(input_ids=input_ids)
    finally:
        handle.remove()

    acts = captured["act"].to(sae.dtype).to(sae.device)  # (1, seq, 768)

    # Encode con SAE
    top_acts, top_indices = sae.encode(acts)  # (1, seq, k)

    # Para el último token
    last_pos = input_ids.shape[1] - 1
    last_acts = top_acts[0, last_pos, :]  # (k,)
    last_indices = top_indices[0, last_pos, :]  # (k,)

    # Dense vector para último token
    dense = torch.zeros(NUM_LATENTS, device=DEVICE)
    dense[last_indices.long()] = last_acts.float()

    # Top features (las k mas activas ya están en top_acts)
    sorted_acts, sorted_order = last_acts.sort(descending=True)
    sorted_indices = last_indices[sorted_order]

    n_show = min(top_k_features, len(sorted_acts))

    lines = [f"## Top {n_show} features activas (último token)\n"]
    lines.append("| Rank | Feature | Activación |")
    lines.append("|------|---------|------------|")
    for i in range(n_show):
        feat_id = sorted_indices[i].item()
        act_val = sorted_acts[i].item()
        lines.append(f"| {i+1} | {feat_id} | {act_val:.4f} |")

    # Mean activation across all tokens
    mean_acts = top_acts[0].float()  # (seq, k)
    mean_indices = top_indices[0]  # (seq, k)
    dense_mean = torch.zeros(NUM_LATENTS, device=DEVICE)
    for t in range(mean_acts.shape[0]):
        dense_mean[mean_indices[t].long()] += mean_acts[t]
    dense_mean /= input_ids.shape[1]

    top_mean_vals, top_mean_ids = dense_mean.topk(n_show)

    lines.append(f"\n## Top {n_show} features activas (media de todos los tokens)\n")
    lines.append("| Rank | Feature | Activación media |")
    lines.append("|------|---------|------------------|")
    for i in range(n_show):
        lines.append(f"| {i+1} | {top_mean_ids[i].item()} | {top_mean_vals[i].item():.4f} |")

    # Plot
    feat_ids = [sorted_indices[i].item() for i in range(n_show)]
    act_vals = [sorted_acts[i].item() for i in range(n_show)]

    fig = go.Figure(go.Bar(
        x=[str(f) for f in feat_ids],
        y=act_vals,
        marker_color="steelblue",
    ))
    fig.update_layout(
        title="Activaciones del último token",
        xaxis_title="Feature ID",
        yaxis_title="Activación",
        height=350,
    )

    # Token-level heatmap para una feature seleccionada (la top 1)
    token_strs = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

    return "\n".join(lines), fig


# =====================
# 3. ANALISIS GENERO - Features discriminativas
# =====================


def compute_gender_discriminative_features(n_sample: int = 200000, top_n: int = 50) -> tuple:
    """Calcula features con mayor diferencia de activacion entre M y F."""
    if gender_acts is None:
        return "No hay activaciones de genero disponibles.", None

    n_total = len(gender_labels)
    rng = np.random.RandomState(42)

    # Sample indices
    f_idx = np.where(gender_labels == 0)[0]
    m_idx = np.where(gender_labels == 1)[0]

    n_per = min(n_sample // 2, len(f_idx), len(m_idx))
    f_sample = rng.choice(f_idx, size=n_per, replace=False)
    m_sample = rng.choice(m_idx, size=n_per, replace=False)

    print(f"Calculando features discriminativas: {n_per:,} F, {n_per:,} M")

    # Media por genero (en chunks para no saturar RAM con float16->float32)
    chunk = 10000
    f_means = np.zeros(gender_num_latents, dtype=np.float64)
    for s in range(0, n_per, chunk):
        e = min(s + chunk, n_per)
        f_means += np.asarray(gender_acts[f_sample[s:e]], dtype=np.float32).sum(axis=0)
    f_means /= n_per

    m_means = np.zeros(gender_num_latents, dtype=np.float64)
    for s in range(0, n_per, chunk):
        e = min(s + chunk, n_per)
        m_means += np.asarray(gender_acts[m_sample[s:e]], dtype=np.float32).sum(axis=0)
    m_means /= n_per

    diff = m_means - f_means  # positivo = más activa en M

    # Top features mas activas en M
    top_m = np.argsort(diff)[-top_n:][::-1]
    # Top features mas activas en F
    top_f = np.argsort(diff)[:top_n]

    lines = [f"## Top {top_n} features más activas en MALE vs FEMALE\n"]
    lines.append("| Rank | Feature | Diff (M-F) | Media M | Media F |")
    lines.append("|------|---------|------------|---------|---------|")
    for i, idx in enumerate(top_m[:20]):
        lines.append(f"| {i+1} | {idx} | {diff[idx]:.6f} | {m_means[idx]:.6f} | {f_means[idx]:.6f} |")

    lines.append(f"\n## Top {top_n} features más activas en FEMALE vs MALE\n")
    lines.append("| Rank | Feature | Diff (F-M) | Media F | Media M |")
    lines.append("|------|---------|------------|---------|---------|")
    for i, idx in enumerate(top_f[:20]):
        lines.append(f"| {i+1} | {idx} | {-diff[idx]:.6f} | {f_means[idx]:.6f} | {m_means[idx]:.6f} |")

    # Plot
    all_top = np.concatenate([top_m[:20], top_f[:20]])
    all_diff = diff[all_top]
    all_labels = [f"F{idx}" for idx in all_top]

    colors = ["#4169E1" if d > 0 else "#DC143C" for d in all_diff]

    fig = go.Figure(go.Bar(
        x=[str(i) for i in all_top],
        y=all_diff,
        marker_color=colors,
        text=[f"{d:.5f}" for d in all_diff],
    ))
    fig.update_layout(
        title="Features discriminativas por género (azul=M, rojo=F)",
        xaxis_title="Feature ID",
        yaxis_title="Diferencia media (M - F)",
        height=500,
    )

    return "\n".join(lines), fig


def get_gender_feature_tokens(feature_idx: int) -> str:
    """Para una feature discriminativa, muestra los tokens asociados."""
    return get_top_tokens_for_feature(feature_idx, top_k=15)


# =====================
# 3b. INTERPRETABILIDAD GENERO - Neuronas de genero
# =====================

# Cache global para no recalcular
_gender_analysis_cache = {}


def _compute_gender_per_feature_stats(n_sample: int = 300000):
    """Calcula estadisticas por feature: media M, media F, Cohen's d,
    correlacion punto-biserial, y accuracy de clasificador de umbral."""
    cache_key = f"gender_stats_{n_sample}"
    if cache_key in _gender_analysis_cache:
        return _gender_analysis_cache[cache_key]

    if gender_acts is None:
        return None

    rng = np.random.RandomState(42)
    f_idx = np.where(gender_labels == 0)[0]
    m_idx = np.where(gender_labels == 1)[0]
    n_per = min(n_sample // 2, len(f_idx), len(m_idx))

    f_sample = rng.choice(f_idx, size=n_per, replace=False)
    m_sample = rng.choice(m_idx, size=n_per, replace=False)

    print(f"Calculando estadisticas por feature: {n_per:,} F, {n_per:,} M ...")
    nl = gender_num_latents
    chunk = 10000

    # Media y varianza por genero (Welford online)
    f_means = np.zeros(nl, dtype=np.float64)
    f_sq = np.zeros(nl, dtype=np.float64)
    for s in range(0, n_per, chunk):
        e = min(s + chunk, n_per)
        batch = np.asarray(gender_acts[f_sample[s:e]], dtype=np.float32)
        f_means += batch.sum(axis=0)
        f_sq += (batch ** 2).sum(axis=0)
    f_means /= n_per
    f_var = f_sq / n_per - f_means ** 2

    m_means = np.zeros(nl, dtype=np.float64)
    m_sq = np.zeros(nl, dtype=np.float64)
    for s in range(0, n_per, chunk):
        e = min(s + chunk, n_per)
        batch = np.asarray(gender_acts[m_sample[s:e]], dtype=np.float32)
        m_means += batch.sum(axis=0)
        m_sq += (batch ** 2).sum(axis=0)
    m_means /= n_per
    m_var = m_sq / n_per - m_means ** 2

    # Cohen's d = (mean_M - mean_F) / pooled_std
    pooled_std = np.sqrt((f_var + m_var) / 2).clip(min=1e-10)
    cohens_d = (m_means - f_means) / pooled_std

    # Correlacion punto-biserial: r = (mean1 - mean0) / s * sqrt(n0*n1/N^2)
    # Simplificada: proporcional a Cohen's d con factor de correccion
    r_pb = cohens_d / np.sqrt(cohens_d ** 2 + 4)  # aproximacion

    # Frecuencia de activacion (% de ejemplos donde la feature se activa > 0)
    all_sample = np.concatenate([f_sample, m_sample])
    freq = np.zeros(nl, dtype=np.float64)
    for s in range(0, len(all_sample), chunk):
        e = min(s + chunk, len(all_sample))
        batch = np.asarray(gender_acts[all_sample[s:e]], dtype=np.float32)
        freq += (batch > 0).sum(axis=0)
    freq /= len(all_sample)

    result = {
        "f_means": f_means, "m_means": m_means,
        "f_var": f_var, "m_var": m_var,
        "cohens_d": cohens_d, "r_pb": r_pb,
        "freq": freq, "n_per": n_per,
    }
    _gender_analysis_cache[cache_key] = result
    print("Estadisticas por feature calculadas.")
    return result


def find_gender_neurons(n_sample: int = 300000, top_n: int = 30) -> tuple:
    """Encuentra las 'neuronas de genero': features SAE con mayor
    poder discriminativo individual para predecir genero.

    Ordena por |Cohen's d| (tamaño del efecto). Un d > 0.2 es
    efecto pequeño, d > 0.5 medio, d > 0.8 grande.
    """
    stats = _compute_gender_per_feature_stats(int(n_sample))
    if stats is None:
        return "No hay activaciones de genero.", None, None

    cohens_d = stats["cohens_d"]
    abs_d = np.abs(cohens_d)
    freq = stats["freq"]
    f_means = stats["f_means"]
    m_means = stats["m_means"]

    # Top por |Cohen's d|
    top_idx = np.argsort(abs_d)[::-1][:top_n]

    lines = [
        "## Neuronas de Genero - Top features por |Cohen's d|\n",
        "Cohen's d mide el tamaño del efecto: cuanto se separan M y F "
        "en activacion de esa feature, normalizado por la variabilidad.\n",
        "- |d| > 0.2: efecto pequeño",
        "- |d| > 0.5: efecto medio",
        "- |d| > 0.8: efecto grande\n",
        "| Rank | Feature | Cohen's d | Dir | Media M | Media F | Freq activ. | Top 3 tokens (logit lens) |",
        "|------|---------|-----------|-----|---------|---------|-------------|---------------------------|",
    ]

    for i, idx in enumerate(top_idx):
        d = cohens_d[idx]
        direction = "M" if d > 0 else "F"

        # Top 3 tokens de logit lens
        dec_vec = W_dec[idx]
        logits = W_U @ dec_vec
        top3_vals, top3_ids = logits.topk(3)
        top3_str = ", ".join(repr(tokenizer.decode([t.item()])) for t in top3_ids)

        lines.append(
            f"| {i+1} | **{idx}** | {d:.4f} | {direction} | "
            f"{m_means[idx]:.5f} | {f_means[idx]:.5f} | {freq[idx]:.3f} | {top3_str} |"
        )

    # Estadisticas generales
    n_small = int((abs_d > 0.2).sum())
    n_medium = int((abs_d > 0.5).sum())
    n_large = int((abs_d > 0.8).sum())
    lines.append(f"\n### Resumen")
    lines.append(f"- Features con |d| > 0.2 (efecto pequeño): **{n_small:,}** / {len(abs_d):,}")
    lines.append(f"- Features con |d| > 0.5 (efecto medio): **{n_medium:,}** / {len(abs_d):,}")
    lines.append(f"- Features con |d| > 0.8 (efecto grande): **{n_large:,}** / {len(abs_d):,}")

    # Plot 1: Bar chart de Cohen's d para top features
    fig_bar = go.Figure(go.Bar(
        x=[str(idx) for idx in top_idx],
        y=[cohens_d[idx] for idx in top_idx],
        marker_color=["#4169E1" if cohens_d[idx] > 0 else "#DC143C" for idx in top_idx],
        text=[f"{cohens_d[idx]:.3f}" for idx in top_idx],
    ))
    fig_bar.update_layout(
        title="Top neuronas de género por Cohen's d (azul=M, rojo=F)",
        xaxis_title="Feature ID", yaxis_title="Cohen's d",
        height=450,
    )

    # Plot 2: Volcano plot (efecto vs frecuencia)
    fig_volcano = go.Figure()
    fig_volcano.add_trace(go.Scattergl(
        x=cohens_d, y=freq,
        mode="markers",
        marker=dict(
            size=3, opacity=0.3,
            color=np.where(abs_d > 0.2, np.where(cohens_d > 0, 0, 1), 2),
            colorscale=[[0, "#4169E1"], [0.5, "#DC143C"], [1, "lightgray"]],
        ),
        text=[str(i) for i in range(len(cohens_d))],
        hovertemplate="Feature %{text}<br>Cohen's d: %{x:.4f}<br>Freq: %{y:.3f}",
    ))
    # Highlight top features
    fig_volcano.add_trace(go.Scattergl(
        x=[cohens_d[i] for i in top_idx[:10]],
        y=[freq[i] for i in top_idx[:10]],
        mode="markers+text",
        marker=dict(size=10, color="red", symbol="star"),
        text=[str(i) for i in top_idx[:10]],
        textposition="top center",
        name="Top 10",
    ))
    fig_volcano.update_layout(
        title="Volcano plot: Cohen's d vs Frecuencia de activación",
        xaxis_title="Cohen's d (+ = más en M, − = más en F)",
        yaxis_title="Frecuencia de activación",
        height=500, showlegend=False,
    )

    return "\n".join(lines), fig_bar, fig_volcano


def analyze_text_gender_features(text: str, n_sample: int = 300000) -> tuple:
    """Para un texto dado, pasa por GPT-2 + SAE y muestra cuales de sus
    64 features activas son 'neuronas de genero'."""
    if not text.strip():
        return "Introduce un texto.", None

    stats = _compute_gender_per_feature_stats(int(n_sample))
    if stats is None:
        return "No hay datos de genero.", None

    cohens_d = stats["cohens_d"]

    # Forward pass
    tokens = tokenizer(
        text, max_length=CONTEXT_LEN, truncation=True,
        padding=False, return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(DEVICE)

    hookpoint = gpt2_model.get_submodule(sae.cfg.hookpoint)
    captured = {}

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        captured["act"] = outputs.detach()

    handle = hookpoint.register_forward_hook(hook)
    try:
        with torch.no_grad():
            gpt2_model.transformer(input_ids=input_ids)
    finally:
        handle.remove()

    acts = captured["act"].to(sae.dtype).to(sae.device)
    top_acts_t, top_indices_t = sae.encode(acts)  # (1, seq, k)

    # Último token
    last_pos = input_ids.shape[1] - 1
    active_ids = top_indices_t[0, last_pos, :].cpu().numpy()  # (k=64,)
    active_vals = top_acts_t[0, last_pos, :].float().cpu().numpy()  # (k=64,)

    # Ordenar las 64 features activas por |Cohen's d|
    active_d = cohens_d[active_ids]
    order = np.argsort(np.abs(active_d))[::-1]

    lines = [
        f"## Análisis de género del texto (último token)\n",
        f"De las **{len(active_ids)}** features activas (top-k={sae.cfg.k}), "
        f"estas son las más relevantes para género:\n",
        "| Rank | Feature | Activación | Cohen's d | Dir | Top 3 tokens |",
        "|------|---------|------------|-----------|-----|---------------|",
    ]

    n_gender_relevant = 0
    for rank, oi in enumerate(order):
        fid = int(active_ids[oi])
        act_val = active_vals[oi]
        d = cohens_d[fid]
        direction = "M" if d > 0 else "F"
        abs_d = abs(d)

        dec_vec = W_dec[fid]
        logits = W_U @ dec_vec
        top3_vals, top3_ids = logits.topk(3)
        top3_str = ", ".join(repr(tokenizer.decode([t.item()])) for t in top3_ids)

        marker = ""
        if abs_d > 0.5:
            marker = " **"
            n_gender_relevant += 1
        elif abs_d > 0.2:
            marker = " *"
            n_gender_relevant += 1

        lines.append(
            f"| {rank+1} | **{fid}**{marker} | {act_val:.4f} | {d:.4f} | {direction} | {top3_str} |"
        )

    lines.append(f"\n**{n_gender_relevant}** de {len(active_ids)} features activas "
                 f"tienen señal de género (|d| > 0.2)")

    vote_m = sum(active_vals[i] * cohens_d[active_ids[i]]
                 for i in range(len(active_ids)) if cohens_d[active_ids[i]] > 0)
    vote_f = sum(active_vals[i] * abs(cohens_d[active_ids[i]])
                 for i in range(len(active_ids)) if cohens_d[active_ids[i]] < 0)

    pred = "MALE" if vote_m > vote_f else "FEMALE"
    lines.append(f"\n### Predicción por votación ponderada")
    lines.append(f"- Evidencia Male: {vote_m:.4f}")
    lines.append(f"- Evidencia Female: {vote_f:.4f}")
    lines.append(f"- **Predicción: {pred}**")

    # Plot: barras coloreadas por dirección de género
    fig = go.Figure(go.Bar(
        x=[str(int(active_ids[order[i]])) for i in range(len(order))],
        y=[active_vals[order[i]] for i in range(len(order))],
        marker_color=[
            "#4169E1" if cohens_d[active_ids[order[i]]] > 0.2
            else ("#DC143C" if cohens_d[active_ids[order[i]]] < -0.2 else "lightgray")
            for i in range(len(order))
        ],
        text=[f"d={cohens_d[active_ids[order[i]]]:.2f}" for i in range(len(order))],
    ))
    fig.update_layout(
        title="64 features activas coloreadas por señal de género (azul=M, rojo=F, gris=neutral)",
        xaxis_title="Feature ID (ordenadas por |Cohen's d|)",
        yaxis_title="Activación",
        height=450,
    )

    return "\n".join(lines), fig


def inspect_gender_neuron(feature_idx: int, n_sample: int = 300000) -> tuple:
    """Inspeccion detallada de una neurona de genero individual."""
    if feature_idx < 0 or feature_idx >= NUM_LATENTS:
        return "Index fuera de rango.", None

    stats = _compute_gender_per_feature_stats(int(n_sample))
    if stats is None:
        return "No hay datos de genero.", None

    d = stats["cohens_d"][feature_idx]
    f_mean = stats["f_means"][feature_idx]
    m_mean = stats["m_means"][feature_idx]
    f_std = np.sqrt(stats["f_var"][feature_idx])
    m_std = np.sqrt(stats["m_var"][feature_idx])
    freq = stats["freq"][feature_idx]

    dec_vec = W_dec[feature_idx]
    logits = W_U @ dec_vec
    top_vals, top_ids = logits.topk(10)
    bot_vals, bot_ids = logits.topk(10, largest=False)

    direction = "MALE" if d > 0 else "FEMALE"
    effect = "pequeño" if abs(d) < 0.5 else ("medio" if abs(d) < 0.8 else "grande")

    lines = [
        f"## Feature {feature_idx} - Neurona de Género\n",
        f"- **Dirección**: más activa en **{direction}**",
        f"- **Cohen's d**: {d:.4f} (efecto {effect})",
        f"- **Media Male**: {m_mean:.6f} ± {m_std:.6f}",
        f"- **Media Female**: {f_mean:.6f} ± {f_std:.6f}",
        f"- **Frecuencia de activación**: {freq:.3f} ({freq*100:.1f}% de comentarios)",
        f"",
        f"### Tokens que esta feature PROMUEVE:",
        "| Rank | Token | Logit |",
        "|------|-------|-------|",
    ]
    for i in range(10):
        tok = repr(tokenizer.decode([top_ids[i].item()]))
        lines.append(f"| {i+1} | {tok} | {top_vals[i].item():.3f} |")

    lines.append(f"\n### Tokens que esta feature SUPRIME:")
    lines.append("| Rank | Token | Logit |")
    lines.append("|------|-------|-------|")
    for i in range(10):
        tok = repr(tokenizer.decode([bot_ids[i].item()]))
        lines.append(f"| {i+1} | {tok} | {bot_vals[i].item():.3f} |")

    # Histograma de activaciones M vs F (sampled)
    rng = np.random.RandomState(42)
    f_idx_all = np.where(gender_labels == 0)[0]
    m_idx_all = np.where(gender_labels == 1)[0]
    n_hist = min(50000, len(f_idx_all), len(m_idx_all))
    f_hist_idx = rng.choice(f_idx_all, size=n_hist, replace=False)
    m_hist_idx = rng.choice(m_idx_all, size=n_hist, replace=False)

    f_vals = np.asarray(gender_acts[f_hist_idx, feature_idx], dtype=np.float32)
    m_vals = np.asarray(gender_acts[m_hist_idx, feature_idx], dtype=np.float32)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=f_vals[f_vals > 0], name="Female", marker_color="#DC143C",
        opacity=0.6, nbinsx=50,
    ))
    fig.add_trace(go.Histogram(
        x=m_vals[m_vals > 0], name="Male", marker_color="#4169E1",
        opacity=0.6, nbinsx=50,
    ))
    fig.update_layout(
        title=f"Feature {feature_idx} - Distribución de activaciones por género",
        xaxis_title="Activación", yaxis_title="Count",
        barmode="overlay", height=400,
    )

    return "\n".join(lines), fig


# =====================
# 4. ESTADISTICAS GLOBALES
# =====================


def compute_global_stats() -> tuple:
    """Estadísticas globales de la SAE."""
    dec_norms = W_dec.norm(dim=1).cpu().numpy()  # (num_latents,)
    enc_norms = W_enc.norm(dim=1).cpu().numpy()
    enc_biases = b_enc.cpu().numpy()

    # Cosine similarity entre encoder y decoder por feature
    cos_sims = F.cosine_similarity(W_enc, W_dec, dim=1).cpu().numpy()

    lines = [
        "## Estadísticas Globales de la SAE\n",
        f"- **Num latents**: {NUM_LATENTS:,}",
        f"- **d_in**: {sae.cfg.d_in}",
        f"- **k (top-k)**: {sae.cfg.k}",
        f"- **Hookpoint**: {sae.cfg.hookpoint}",
        f"",
        f"### Decoder weight norms",
        f"- Media: {dec_norms.mean():.4f} ± {dec_norms.std():.4f}",
        f"- Min: {dec_norms.min():.4f}, Max: {dec_norms.max():.4f}",
        f"",
        f"### Encoder weight norms",
        f"- Media: {enc_norms.mean():.4f} ± {enc_norms.std():.4f}",
        f"",
        f"### Encoder biases",
        f"- Media: {enc_biases.mean():.4f} ± {enc_biases.std():.4f}",
        f"- Min: {enc_biases.min():.4f}, Max: {enc_biases.max():.4f}",
        f"- % negativos (dead features proxy): {100*(enc_biases < -5).sum()/len(enc_biases):.1f}%",
        f"",
        f"### Cosine similarity (enc ↔ dec)",
        f"- Media: {cos_sims.mean():.4f} ± {cos_sims.std():.4f}",
    ]

    # Distribución de normas
    fig1 = px.histogram(
        x=dec_norms, nbins=100,
        title="Distribución de decoder weight norms",
        labels={"x": "Norma", "y": "Count"},
    )
    fig1.update_layout(height=350)

    # Distribución de biases
    fig2 = px.histogram(
        x=enc_biases, nbins=100,
        title="Distribución de encoder biases",
        labels={"x": "Bias", "y": "Count"},
    )
    fig2.update_layout(height=350)

    # Cosine similarity distribution
    fig3 = px.histogram(
        x=cos_sims, nbins=100,
        title="Cosine similarity entre encoder y decoder weights",
        labels={"x": "Cosine sim", "y": "Count"},
    )
    fig3.update_layout(height=350)

    return "\n".join(lines), fig1, fig2, fig3


# =====================
# 5. FEATURE SIMILARITY
# =====================


def find_similar_features(feature_idx: int, top_k: int = 10) -> str:
    """Encuentra las features mas similares (por cosine sim del decoder weight)."""
    if feature_idx < 0 or feature_idx >= NUM_LATENTS:
        return "Index fuera de rango"

    dec_vec = W_dec[feature_idx].unsqueeze(0)  # (1, d_in)
    sims = F.cosine_similarity(dec_vec, W_dec, dim=1)  # (num_latents,)
    sims[feature_idx] = -1  # excluir self

    top_vals, top_ids = sims.topk(top_k)

    lines = [f"## Features más similares a Feature {feature_idx}\n"]
    lines.append("| Rank | Feature | Cosine Sim |")
    lines.append("|------|---------|------------|")
    for i in range(top_k):
        lines.append(f"| {i+1} | {top_ids[i].item()} | {top_vals[i].item():.4f} |")

    return "\n".join(lines)


# =====================
# 6. TOKEN-LEVEL FEATURE ACTIVATION MAP
# =====================


def token_feature_heatmap(text: str, feature_idx: int):
    """Muestra la activación de una feature específica en cada token del texto."""
    if not text.strip():
        return "Introduce un texto.", None
    if feature_idx < 0 or feature_idx >= NUM_LATENTS:
        return "Feature index fuera de rango.", None

    tokens = tokenizer(
        text, max_length=CONTEXT_LEN, truncation=True,
        padding=False, return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(DEVICE)

    hookpoint = gpt2_model.get_submodule(sae.cfg.hookpoint)
    captured = {}

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        captured["act"] = outputs.detach()

    handle = hookpoint.register_forward_hook(hook)
    try:
        with torch.no_grad():
            gpt2_model.transformer(input_ids=input_ids)
    finally:
        handle.remove()

    acts = captured["act"].to(sae.dtype).to(sae.device)
    top_acts, top_indices = sae.encode(acts)  # (1, seq, k)

    seq_len = input_ids.shape[1]
    token_strs = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

    # Extraer activación de la feature específica en cada posición
    feature_activation = np.zeros(seq_len)
    for t in range(seq_len):
        mask = (top_indices[0, t] == feature_idx)
        if mask.any():
            feature_activation[t] = top_acts[0, t][mask].float().sum().item()

    # Crear heatmap
    fig = go.Figure(go.Bar(
        x=list(range(seq_len)),
        y=feature_activation,
        text=token_strs,
        textposition="outside",
        marker_color=["red" if v > 0 else "lightgray" for v in feature_activation],
    ))
    fig.update_layout(
        title=f"Activación de Feature {feature_idx} por token",
        xaxis_title="Posición",
        yaxis_title="Activación",
        height=400,
        xaxis=dict(tickmode="array", tickvals=list(range(seq_len)), ticktext=token_strs, tickangle=45),
    )

    active_tokens = [(token_strs[i], feature_activation[i]) for i in range(seq_len) if feature_activation[i] > 0]
    if active_tokens:
        lines = [f"## Feature {feature_idx} - Tokens activos\n"]
        lines.append("| Token | Activación |")
        lines.append("|-------|------------|")
        for tok, val in sorted(active_tokens, key=lambda x: -x[1]):
            lines.append(f"| {repr(tok)} | {val:.4f} |")
    else:
        lines = [f"Feature {feature_idx} no se activa en este texto."]

    return "\n".join(lines), fig


# =====================
# GRADIO UI
# =====================


def build_app():
    with gr.Blocks(title="SAE Dashboard - GPT-2 Comments", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔬 SAE Dashboard - Análisis de Features")
        gr.Markdown(f"**SAE**: {NUM_LATENTS:,} latentes, k={sae.cfg.k}, hookpoint={sae.cfg.hookpoint}")

        with gr.Tabs():
            # ---- Tab 1: Feature Browser ----
            with gr.Tab("Feature Browser"):
                gr.Markdown("Explora features individuales: tokens asociados (logit lens), estadísticas, features similares.")
                with gr.Row():
                    feat_input = gr.Number(label="Feature ID", value=0, precision=0, minimum=0, maximum=NUM_LATENTS-1)
                    top_k_input = gr.Number(label="Top K tokens", value=20, precision=0, minimum=5, maximum=50)
                with gr.Row():
                    btn_tokens = gr.Button("Ver Top Tokens (Logit Lens)")
                    btn_stats = gr.Button("Estadísticas")
                    btn_similar = gr.Button("Features Similares")
                tokens_output = gr.Markdown()
                stats_output = gr.Markdown()
                hist_output = gr.Plot()
                similar_output = gr.Markdown()

                btn_tokens.click(get_top_tokens_for_feature, [feat_input, top_k_input], tokens_output)
                btn_stats.click(get_feature_stats, [feat_input], stats_output)
                btn_stats.click(plot_feature_decoder_histogram, [feat_input], hist_output)
                btn_similar.click(find_similar_features, [feat_input], similar_output)

            # ---- Tab 2: Analizar Texto ----
            with gr.Tab("Analizar Texto"):
                gr.Markdown("Pasa un texto por GPT-2 + SAE y ve qué features se activan.")
                text_input = gr.Textbox(label="Texto", lines=4, placeholder="Escribe un comentario aquí...")
                top_k_feat = gr.Number(label="Top K features", value=15, precision=0)
                btn_analyze = gr.Button("Analizar")
                text_result = gr.Markdown()
                text_plot = gr.Plot()
                btn_analyze.click(analyze_text, [text_input, top_k_feat], [text_result, text_plot])

                gr.Markdown("---")
                gr.Markdown("### Heatmap por token de una feature específica")
                with gr.Row():
                    hm_text = gr.Textbox(label="Texto", lines=2, placeholder="Texto corto...")
                    hm_feat = gr.Number(label="Feature ID", value=0, precision=0)
                btn_heatmap = gr.Button("Ver Heatmap")
                hm_result = gr.Markdown()
                hm_plot = gr.Plot()
                btn_heatmap.click(token_feature_heatmap, [hm_text, hm_feat], [hm_result, hm_plot])

            # ---- Tab 3: Análisis Género ----
            with gr.Tab("Análisis Género"):
                gr.Markdown("Features con mayor diferencia de activación entre géneros (datos SAE de género).")
                with gr.Row():
                    n_sample = gr.Number(label="N muestras por género", value=200000, precision=0)
                    n_top = gr.Number(label="Top N features", value=50, precision=0)
                btn_gender = gr.Button("Calcular Features Discriminativas")
                gender_result = gr.Markdown()
                gender_plot = gr.Plot()
                btn_gender.click(compute_gender_discriminative_features, [n_sample, n_top], [gender_result, gender_plot])

                gr.Markdown("---")
                gr.Markdown("### Inspeccionar feature discriminativa")
                gender_feat = gr.Number(label="Feature ID", value=0, precision=0)
                btn_gender_tokens = gr.Button("Ver tokens de esta feature")
                gender_tokens_output = gr.Markdown()
                btn_gender_tokens.click(get_gender_feature_tokens, [gender_feat], gender_tokens_output)

            # ---- Tab 4: Interpretabilidad Género ----
            with gr.Tab("Neuronas de Género"):
                gr.Markdown(
                    "Análisis de interpretabilidad: ¿qué features SAE individuales (de las 16384) "
                    "actúan como 'neuronas de género'? Ordenadas por Cohen's d (tamaño del efecto)."
                )
                with gr.Row():
                    gn_nsample = gr.Number(label="N muestras por género", value=300000, precision=0)
                    gn_topn = gr.Number(label="Top N neuronas", value=30, precision=0)
                btn_neurons = gr.Button("Buscar Neuronas de Género")
                gn_result = gr.Markdown()
                gn_fig_bar = gr.Plot()
                gn_fig_volcano = gr.Plot()
                btn_neurons.click(find_gender_neurons, [gn_nsample, gn_topn], [gn_result, gn_fig_bar, gn_fig_volcano])

                gr.Markdown("---")
                gr.Markdown("### Inspeccionar una neurona de género en detalle")
                gr.Markdown("Muestra tokens asociados, estadísticas, y distribución de activaciones M vs F.")
                gn_feat_id = gr.Number(label="Feature ID", value=0, precision=0)
                btn_inspect = gr.Button("Inspeccionar Neurona")
                gn_inspect_result = gr.Markdown()
                gn_inspect_fig = gr.Plot()
                btn_inspect.click(inspect_gender_neuron, [gn_feat_id, gn_nsample], [gn_inspect_result, gn_inspect_fig])

                gr.Markdown("---")
                gr.Markdown("### Analizar un texto: ¿qué neuronas de género se activan?")
                gr.Markdown(
                    "Introduce un comentario y ve cuáles de sus 64 features activas (top-k) "
                    "son neuronas de género y hacia qué dirección empujan."
                )
                gn_text = gr.Textbox(label="Texto", lines=3, placeholder="Escribe un comentario...")
                btn_text_gender = gr.Button("Analizar Señal de Género")
                gn_text_result = gr.Markdown()
                gn_text_fig = gr.Plot()
                btn_text_gender.click(analyze_text_gender_features, [gn_text, gn_nsample], [gn_text_result, gn_text_fig])

            # ---- Tab 5: Estadísticas Globales ----
            with gr.Tab("Estadísticas Globales"):
                btn_global = gr.Button("Calcular Estadísticas")
                global_result = gr.Markdown()
                global_fig1 = gr.Plot()
                global_fig2 = gr.Plot()
                global_fig3 = gr.Plot()
                btn_global.click(compute_global_stats, [], [global_result, global_fig1, global_fig2, global_fig3])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
