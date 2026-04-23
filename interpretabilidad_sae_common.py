from __future__ import annotations

import heapq
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from preprocesamiento import preparar_dataset_para_edad, preparar_dataset_para_mbti, preparar_dataset_para_sae
from tiny_sae import Sae


MODEL = "openai-community/gpt2"
PATH_SAE = "sae-ckpts/sae-gpt2-comments"
PATH_COMENTARIOS = "data/all_comments_since_2015.csv"
PATH_AUTORES = "data/author_profiles.csv"
TEXT_COLUMN = "body"
CONTEXT_LEN = 512
EXTRACT_BATCH_SIZE = 32
MIN_EXTRACT_BATCH_SIZE = 4
PROGRESS_INTERVAL = 3600
RANDOM_STATE = 42
SGD_ALPHA = 1e-5
TOP_K_LATENTS = 20
TOP_EXAMPLES_PER_LATENT = 8
ABLATION_SIZES = (5, 10, 20)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

SAE_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

STOPWORDS = {
    "about", "after", "again", "against", "all", "also", "always", "among", "and", "any",
    "are", "around", "because", "been", "before", "being", "both", "but", "can", "cant",
    "could", "couldnt", "did", "didnt", "does", "doesnt", "doing", "dont", "each", "even",
    "every", "from", "get", "gets", "getting", "got", "had", "has", "have", "having", "her",
    "here", "hers", "him", "his", "how", "http", "https", "into", "ive", "just", "like",
    "more", "most", "much", "must", "myself", "not", "now", "off", "onto", "other", "our",
    "ours", "out", "over", "reddit", "really", "same", "she", "should", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "this", "those",
    "through", "too", "under", "until", "very", "was", "wasnt", "were", "werent", "what", "when",
    "where", "which", "while", "who", "will", "with", "within", "without", "would", "wouldnt", "you",
    "your", "yours", "youre", "its", "ill", "amp", "www", "com", "org", "net",
}


@dataclass(frozen=True)
class SaeInterpretabilityConfig:
    task_name: str
    dataset_kind: str
    splits_dir: str
    output_dir: str
    class_names: Tuple[str, ...]
    cache_dir: Optional[str] = None
    mbti_trait: Optional[str] = None
    age_groups: Optional[Tuple[str, ...]] = None
    max_comments: Optional[int] = None
    top_k_latents: int = TOP_K_LATENTS
    top_examples_per_latent: int = TOP_EXAMPLES_PER_LATENT
    ablation_sizes: Tuple[int, ...] = ABLATION_SIZES


def run_posthoc_analysis(config: SaeInterpretabilityConfig) -> Dict[str, object]:
    if not Path(PATH_SAE).exists():
        raise FileNotFoundError(f"No se encuentra la SAE en {PATH_SAE}")

    df = _load_dataframe(config)
    train_auth, eval_auth, _test_auth = _load_user_splits(config)

    if "author" not in df.columns:
        raise ValueError("El dataset no contiene columna 'author'.")

    df["author"] = df["author"].astype(str).str.strip()
    train_set = set(train_auth)
    eval_set = set(eval_auth)

    df_train = df[df["author"].isin(train_set)].reset_index(drop=True)
    df_eval = df[df["author"].isin(eval_set)].reset_index(drop=True)

    print(f"Train comments: {len(df_train):,} | Eval comments: {len(df_eval):,}")

    tokenizer, model, sae, hookpoint_module, num_latents = _setup_models()
    print(f"num_latents={num_latents}")

    train_users = _aggregate_user_features(
        df=df_train,
        tokenizer=tokenizer,
        model=model,
        sae=sae,
        hookpoint_module=hookpoint_module,
        num_latents=num_latents,
        pass_name="TRAIN",
    )

    X_train, y_train = _build_user_arrays(train_users, num_latents)
    scaler, clf = _fit_user_model(X_train, y_train, len(config.class_names))
    top_latents = _select_top_latents(clf, config.class_names, config.top_k_latents)
    latent_stats = _compute_latent_stats(X_train, y_train, top_latents, config.class_names, clf.coef_)

    eval_users, examples = _aggregate_eval_and_collect_examples(
        df=df_eval,
        tokenizer=tokenizer,
        model=model,
        sae=sae,
        hookpoint_module=hookpoint_module,
        num_latents=num_latents,
        top_latents=top_latents,
        class_names=config.class_names,
        top_examples=config.top_examples_per_latent,
    )

    del model, tokenizer, sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    X_eval, y_eval = _build_user_arrays(eval_users, num_latents)
    metrics, y_pred = _evaluate_user_model(clf, scaler, X_eval, y_eval, config.class_names)
    ablation = _run_ablation(
        X_eval=X_eval,
        y_eval=y_eval,
        scaler=scaler,
        clf=clf,
        top_latents=top_latents,
        class_names=config.class_names,
        num_latents=num_latents,
        baseline_metrics=metrics,
        ablation_sizes=config.ablation_sizes,
    )

    results = {
        "task_name": config.task_name,
        "storage_mode": "streaming_without_feature_cache",
        "class_names": list(config.class_names),
        "num_comments_train": int(len(df_train)),
        "num_comments_eval": int(len(df_eval)),
        "num_users_train": int(len(X_train)),
        "num_users_eval": int(len(X_eval)),
        "num_latents": int(num_latents),
        "pooling": "mean_of_mean",
        "metrics": metrics,
        "ablation": ablation,
        "top_latents_by_class": _format_top_latents(
            config.class_names,
            top_latents,
            latent_stats,
            examples,
        ),
        "num_eval_predictions": int(len(y_pred)),
    }

    os.makedirs(config.output_dir, exist_ok=True)
    json_path = os.path.join(config.output_dir, "interpretabilidad_sae_resumen.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    md_path = os.path.join(config.output_dir, "interpretabilidad_sae_resumen.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(_render_markdown(results))

    print(f"Resumen JSON guardado en: {json_path}")
    print(f"Resumen Markdown guardado en: {md_path}")
    return results


def _load_dataframe(config: SaeInterpretabilityConfig) -> pd.DataFrame:
    if config.dataset_kind == "gender":
        df, _ = preparar_dataset_para_sae(
            path_comentarios=PATH_COMENTARIOS,
            path_autores=PATH_AUTORES,
            max_comments=config.max_comments,
            solo_genero_conocido=True,
        )
        if TEXT_COLUMN in df.columns and TEXT_COLUMN != "text":
            df = df.rename(columns={TEXT_COLUMN: "text"})
        df = df.dropna(subset=["text", "gender_clean", "author"]).reset_index(drop=True)
        df = df[df["gender_clean"].isin(["f", "m"])].reset_index(drop=True)
        df["label"] = np.where(df["gender_clean"] == "f", 0, 1).astype(np.int8)
        return df

    if config.dataset_kind == "age":
        age_groups = config.age_groups or tuple()
        df, _ = preparar_dataset_para_edad(
            path_comentarios=PATH_COMENTARIOS,
            path_autores=PATH_AUTORES,
            max_comments=config.max_comments,
            solo_edad_conocida=True,
        )
        if TEXT_COLUMN in df.columns and TEXT_COLUMN != "text":
            df = df.rename(columns={TEXT_COLUMN: "text"})
        df = df.dropna(subset=["text", "age_group", "author"]).reset_index(drop=True)
        df = df[df["age_group"].isin(age_groups)].reset_index(drop=True)
        label_map = {group: idx for idx, group in enumerate(age_groups)}
        df["label"] = df["age_group"].map(label_map).astype(np.int8)
        return df

    if config.dataset_kind == "mbti":
        if not config.mbti_trait:
            raise ValueError("mbti_trait es obligatorio para dataset_kind='mbti'.")
        df, _ = preparar_dataset_para_mbti(
            path_comentarios=PATH_COMENTARIOS,
            path_autores=PATH_AUTORES,
            columna_mbti=config.mbti_trait,
            max_comments=config.max_comments,
        )
        if TEXT_COLUMN in df.columns and TEXT_COLUMN != "text":
            df = df.rename(columns={TEXT_COLUMN: "text"})
        df = df.dropna(subset=["text", config.mbti_trait, "author"]).reset_index(drop=True)
        df["label"] = df[config.mbti_trait].astype(np.int8)
        return df

    raise ValueError(f"dataset_kind no soportado: {config.dataset_kind}")


def _load_user_splits(config: SaeInterpretabilityConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_npz = os.path.join(config.splits_dir, "split_usuarios.npz")
    if os.path.exists(split_npz):
        data = np.load(split_npz, allow_pickle=True)
        return data["train_auth"], data["eval_auth"], data["test_auth"]

    train_path = os.path.join(config.splits_dir, "usuario_train_auth.npy")
    eval_path = os.path.join(config.splits_dir, "usuario_eval_auth.npy")
    test_path = os.path.join(config.splits_dir, "usuario_test_auth.npy")
    if all(os.path.exists(path) for path in (train_path, eval_path, test_path)):
        return (
            np.load(train_path, allow_pickle=True),
            np.load(eval_path, allow_pickle=True),
            np.load(test_path, allow_pickle=True),
        )

    raise FileNotFoundError(f"No se encontraron splits de usuario reutilizables en {config.splits_dir}")


def _setup_models() -> Tuple[AutoTokenizer, AutoModelForCausalLM, Sae, torch.nn.Module, int]:
    sae = Sae.load_from_disk(PATH_SAE, device=DEVICE)
    if hasattr(sae, "to"):
        sae = sae.to(device=DEVICE, dtype=SAE_DTYPE)
    num_latents = sae.cfg.num_latents
    hookpoint_name = sae.cfg.hookpoint
    print(f"SAE cargada: {num_latents} latentes, k={sae.cfg.k}, hookpoint={hookpoint_name}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": DEVICE},
        dtype=SAE_DTYPE,
    )
    model.eval()

    hookpoint_module = model.get_submodule(hookpoint_name)
    return tokenizer, model, sae, hookpoint_module, num_latents


def _pool_sparse_to_dense(
    top_acts: torch.Tensor,
    top_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    num_latents: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, k = top_acts.shape
    device = top_acts.device
    dtype = top_acts.dtype
    mask = attention_mask.to(device=device, dtype=dtype)

    lengths = mask.sum(dim=1).clamp(min=1).long() - 1
    batch_idx = torch.arange(batch_size, device=device)
    last_acts = top_acts[batch_idx, lengths, :]
    last_indices = top_indices[batch_idx, lengths, :]

    last_pooled = torch.zeros(batch_size, num_latents, device=device, dtype=dtype)
    last_pooled.scatter_add_(1, last_indices.long(), last_acts)

    mean_pooled = torch.zeros(batch_size, num_latents, device=device, dtype=dtype)
    token_mask = mask.unsqueeze(-1)
    masked_acts = top_acts * token_mask
    batch_ids = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, seq_len, k)
    mean_pooled.index_put_(
        (batch_ids.reshape(-1), top_indices.reshape(-1).long()),
        masked_acts.reshape(-1).to(mean_pooled.dtype),
        accumulate=True,
    )
    valid_tokens = mask.sum(dim=1).clamp(min=1).unsqueeze(1)
    mean_pooled = mean_pooled / valid_tokens
    return last_pooled, mean_pooled


def _stream_sae_features(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    sae: Sae,
    hookpoint_module: torch.nn.Module,
    num_latents: int,
    pass_name: str,
):
    n = len(df)
    captured: Dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, outputs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        captured["act"] = outputs.detach()

    handle = hookpoint_module.register_forward_hook(hook)
    step = 0
    total_steps = max(1, math.ceil(n / EXTRACT_BATCH_SIZE))
    current_batch_size = EXTRACT_BATCH_SIZE
    last_print = time.time()

    try:
        with torch.inference_mode():
            start = 0
            while start < n:
                batch_size = current_batch_size
                while True:
                    end = min(start + batch_size, n)
                    batch_texts = df["text"].iloc[start:end].astype(str).tolist()
                    try:
                        tokens = tokenizer(
                            batch_texts,
                            max_length=CONTEXT_LEN,
                            truncation=True,
                            padding="max_length",
                            return_attention_mask=True,
                            return_tensors="pt",
                        )
                        input_ids = tokens["input_ids"].to(model.device)
                        attention_mask = tokens["attention_mask"].to(model.device)
                        model.transformer(input_ids=input_ids, attention_mask=attention_mask)

                        acts = captured["act"].to(sae.dtype).to(sae.device)
                        top_acts, top_indices = sae.encode(acts)
                        _last_pooled, mean_pooled = _pool_sparse_to_dense(
                            top_acts=top_acts,
                            top_indices=top_indices,
                            attention_mask=attention_mask,
                            num_latents=num_latents,
                        )
                        mean_np = mean_pooled.float().cpu().numpy()
                        del tokens, input_ids, attention_mask, acts, top_acts, top_indices, _last_pooled, mean_pooled
                        captured.clear()
                        break
                    except Exception as exc:
                        if not _is_oom_error(exc):
                            raise
                        captured.clear()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if batch_size <= MIN_EXTRACT_BATCH_SIZE:
                            raise RuntimeError(
                                f"OOM incluso con batch_size={batch_size} en pass {pass_name}."
                            ) from exc
                        new_batch_size = max(MIN_EXTRACT_BATCH_SIZE, batch_size // 2)
                        print(f"  OOM: batch {batch_size} -> {new_batch_size} en {pass_name}")
                        batch_size = new_batch_size

                step += 1
                current_batch_size = batch_size
                now = time.time()
                if now - last_print >= PROGRESS_INTERVAL or step == 1 or end >= n:
                    pct = 100.0 * end / max(1, n)
                    print(f"  [{pass_name} {pct:5.1f}%] step {step}/{total_steps} ({end:,}/{n:,}) | batch={current_batch_size}")
                    last_print = now

                yield start, end, mean_np
                start = end
    finally:
        handle.remove()
        captured.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _aggregate_user_features(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    sae: Sae,
    hookpoint_module: torch.nn.Module,
    num_latents: int,
    pass_name: str,
) -> Dict[str, List[object]]:
    user_dict: Dict[str, List[object]] = {}
    for start, end, mean_np in _stream_sae_features(df, tokenizer, model, sae, hookpoint_module, num_latents, pass_name):
        authors = df["author"].iloc[start:end].tolist()
        labels = df["label"].iloc[start:end].to_numpy(dtype=np.int64)
        for idx, author in enumerate(authors):
            if author not in user_dict:
                user_dict[author] = [np.zeros(num_latents, dtype=np.float32), 0, int(labels[idx])]
            entry = user_dict[author]
            entry[0] += mean_np[idx].astype(np.float32, copy=False)
            entry[1] += 1
    return user_dict


def _aggregate_eval_and_collect_examples(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    sae: Sae,
    hookpoint_module: torch.nn.Module,
    num_latents: int,
    top_latents: Dict[int, List[int]],
    class_names: Sequence[str],
    top_examples: int,
) -> Tuple[Dict[str, List[object]], Dict[int, Dict[int, Dict[str, object]]]]:
    user_dict: Dict[str, List[object]] = {}
    heaps: Dict[int, Dict[int, List[Tuple[float, str]]]] = {
        class_idx: {latent: [] for latent in latents}
        for class_idx, latents in top_latents.items()
    }

    for start, end, mean_np in _stream_sae_features(df, tokenizer, model, sae, hookpoint_module, num_latents, "EVAL"):
        authors = df["author"].iloc[start:end].tolist()
        labels = df["label"].iloc[start:end].to_numpy(dtype=np.int64)
        texts = df["text"].iloc[start:end].astype(str).tolist()

        for idx, author in enumerate(authors):
            if author not in user_dict:
                user_dict[author] = [np.zeros(num_latents, dtype=np.float32), 0, int(labels[idx])]
            entry = user_dict[author]
            entry[0] += mean_np[idx].astype(np.float32, copy=False)
            entry[1] += 1

        for class_idx, latents in top_latents.items():
            class_rows = np.where(labels == class_idx)[0]
            if class_rows.size == 0:
                continue
            class_values = mean_np[class_rows][:, latents]
            for latent_pos, latent in enumerate(latents):
                latent_values = class_values[:, latent_pos]
                if latent_values.size == 0:
                    continue
                take_n = min(top_examples, latent_values.size)
                candidate_rows = np.argpartition(latent_values, -take_n)[-take_n:]
                heap = heaps[class_idx][latent]
                for candidate in candidate_rows:
                    activation = float(latent_values[candidate])
                    if activation <= 0:
                        continue
                    text = _truncate_text(texts[int(class_rows[candidate])])
                    item = (activation, text)
                    if len(heap) < top_examples:
                        heapq.heappush(heap, item)
                    elif activation > heap[0][0]:
                        heapq.heapreplace(heap, item)

    examples: Dict[int, Dict[int, Dict[str, object]]] = {class_idx: {} for class_idx in range(len(class_names))}
    for class_idx, latent_heaps in heaps.items():
        for latent, heap in latent_heaps.items():
            ranked = sorted(heap, reverse=True)
            texts = [text for _activation, text in ranked]
            examples[class_idx][latent] = {
                "top_words": _top_words(texts),
                "examples": [
                    {"activation": float(activation), "text": text}
                    for activation, text in ranked
                ],
            }
    return user_dict, examples


def _build_user_arrays(user_dict: Dict[str, List[object]], num_latents: int) -> Tuple[np.ndarray, np.ndarray]:
    users = sorted(user_dict.keys())
    X = np.zeros((len(users), num_latents), dtype=np.float32)
    y = np.zeros(len(users), dtype=np.int64)
    for idx, author in enumerate(users):
        sums, count, label = user_dict[author]
        X[idx] = sums / max(int(count), 1)
        y[idx] = int(label)
    return X, y


def _fit_user_model(X_train: np.ndarray, y_train: np.ndarray, num_classes: int) -> Tuple[StandardScaler, SGDClassifier]:
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    class_weights = _balanced_class_weights(y_train, num_classes)
    sample_weights = class_weights[y_train]

    clf = SGDClassifier(
        loss="log_loss",
        alpha=SGD_ALPHA,
        max_iter=2000,
        tol=1e-3,
        random_state=RANDOM_STATE,
        average=True,
    )
    clf.fit(X_train_n, y_train, sample_weight=sample_weights)
    return scaler, clf


def _evaluate_user_model(
    clf: SGDClassifier,
    scaler: StandardScaler,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    class_names: Sequence[str],
) -> Tuple[Dict[str, float], np.ndarray]:
    X_eval_n = scaler.transform(X_eval)
    y_pred = clf.predict(X_eval_n)
    return _evaluate(y_eval, y_pred, class_names), y_pred


def _balanced_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    weights = np.ones(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (num_classes * counts[nonzero])
    return weights


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str]) -> Dict[str, float]:
    labels = list(range(len(class_names)))
    precision = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    for idx, class_name in enumerate(class_names):
        result[f"precision_{class_name}"] = float(precision[idx])
        result[f"recall_{class_name}"] = float(recall[idx])
        result[f"f1_{class_name}"] = float(f1[idx])
    return result


def _select_top_latents(clf: SGDClassifier, class_names: Sequence[str], top_k: int) -> Dict[int, List[int]]:
    coef = clf.coef_
    top_latents: Dict[int, List[int]] = {}
    if coef.shape[0] == 1 and len(class_names) == 2:
        binary_coef = coef[0]
        top_latents[0] = np.argsort(binary_coef)[:top_k].tolist()
        top_latents[1] = np.argsort(binary_coef)[-top_k:][::-1].tolist()
        return top_latents

    for class_idx in range(len(class_names)):
        top_latents[class_idx] = np.argsort(coef[class_idx])[-top_k:][::-1].tolist()
    return top_latents


def _compute_latent_stats(
    X_train: np.ndarray,
    y_train: np.ndarray,
    top_latents: Dict[int, List[int]],
    class_names: Sequence[str],
    coef: np.ndarray,
) -> Dict[int, Dict[int, Dict[str, object]]]:
    stats: Dict[int, Dict[int, Dict[str, object]]] = {class_idx: {} for class_idx in range(len(class_names))}
    for class_idx, latents in top_latents.items():
        for latent in latents:
            mean_by_class = {}
            nonzero_by_class = {}
            for other_idx, other_name in enumerate(class_names):
                mask = y_train == other_idx
                if mask.any():
                    values = X_train[mask, latent]
                    mean_by_class[other_name] = float(values.mean())
                    nonzero_by_class[other_name] = float((values > 0).mean())
                else:
                    mean_by_class[other_name] = 0.0
                    nonzero_by_class[other_name] = 0.0

            raw_coef = float(coef[0, latent]) if coef.shape[0] == 1 else float(coef[class_idx, latent])
            class_score = -raw_coef if coef.shape[0] == 1 and class_idx == 0 else raw_coef
            stats[class_idx][latent] = {
                "raw_coefficient": raw_coef,
                "class_score": float(class_score),
                "mean_activation_by_class": mean_by_class,
                "nonzero_rate_by_class": nonzero_by_class,
            }
    return stats


def _run_ablation(
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    scaler: StandardScaler,
    clf: SGDClassifier,
    top_latents: Dict[int, List[int]],
    class_names: Sequence[str],
    num_latents: int,
    baseline_metrics: Dict[str, float],
    ablation_sizes: Sequence[int],
) -> Dict[str, object]:
    rng = np.random.RandomState(RANDOM_STATE)
    ablations = {}
    for size in ablation_sizes:
        selected = sorted({latent for latents in top_latents.values() for latent in latents[:size]})
        if not selected:
            continue

        X_zero = X_eval.copy()
        X_zero[:, selected] = 0.0
        y_zero = clf.predict(scaler.transform(X_zero))
        zero_metrics = _evaluate(y_eval, y_zero, class_names)

        random_latents = sorted(rng.choice(num_latents, size=len(selected), replace=False).tolist())
        X_random = X_eval.copy()
        X_random[:, random_latents] = 0.0
        y_random = clf.predict(scaler.transform(X_random))
        random_metrics = _evaluate(y_eval, y_random, class_names)

        ablations[str(size)] = {
            "selected_latents": selected,
            "selected_count": int(len(selected)),
            "metrics_after_zeroing": zero_metrics,
            "metrics_after_random_zeroing": random_metrics,
            "balanced_accuracy_drop": float(baseline_metrics["balanced_accuracy"] - zero_metrics["balanced_accuracy"]),
            "balanced_accuracy_drop_random": float(
                baseline_metrics["balanced_accuracy"] - random_metrics["balanced_accuracy"]
            ),
            "f1_macro_drop": float(baseline_metrics["f1_macro"] - zero_metrics["f1_macro"]),
            "f1_macro_drop_random": float(baseline_metrics["f1_macro"] - random_metrics["f1_macro"]),
        }

    return {
        "baseline_metrics": baseline_metrics,
        "ablations": ablations,
    }


def _format_top_latents(
    class_names: Sequence[str],
    top_latents: Dict[int, List[int]],
    latent_stats: Dict[int, Dict[int, Dict[str, object]]],
    examples: Dict[int, Dict[int, Dict[str, object]]],
) -> Dict[str, List[Dict[str, object]]]:
    formatted = {}
    for class_idx, class_name in enumerate(class_names):
        entries = []
        for latent in top_latents[class_idx]:
            stats = latent_stats[class_idx][latent]
            example_info = examples[class_idx].get(latent, {"top_words": [], "examples": []})
            entries.append(
                {
                    "latent_id": int(latent),
                    "class_score": float(stats["class_score"]),
                    "raw_coefficient": float(stats["raw_coefficient"]),
                    "mean_activation_by_class": stats["mean_activation_by_class"],
                    "nonzero_rate_by_class": stats["nonzero_rate_by_class"],
                    "top_words": example_info["top_words"],
                    "examples": example_info["examples"],
                }
            )
        formatted[class_name] = entries
    return formatted


def _render_markdown(results: Dict[str, object]) -> str:
    lines = []
    lines.append(f"# Interpretabilidad SAE - {results['task_name']}")
    lines.append("")
    lines.append(f"Modo de almacenamiento: {results['storage_mode']}")
    lines.append(f"Pooling analizado: {results['pooling']}")
    lines.append(f"Comentarios train: {results['num_comments_train']:,} | eval: {results['num_comments_eval']:,}")
    lines.append(f"Usuarios train: {results['num_users_train']:,} | eval: {results['num_users_eval']:,}")
    lines.append(f"Latentes SAE: {results['num_latents']:,}")
    lines.append("")

    metrics = results["metrics"]
    lines.append(
        f"Accuracy={metrics['accuracy']:.4f} | Balanced Acc={metrics['balanced_accuracy']:.4f} | "
        f"F1 macro={metrics['f1_macro']:.4f}"
    )
    lines.append("")

    for class_name, latent_entries in results["top_latents_by_class"].items():
        lines.append(f"## Clase {class_name}")
        lines.append("")
        for entry in latent_entries[:10]:
            words = ", ".join(word["token"] for word in entry["top_words"][:8])
            lines.append(
                f"- Latente {entry['latent_id']}: score={entry['class_score']:.4f}, "
                f"coef={entry['raw_coefficient']:.4f}, palabras={words}"
            )
        lines.append("")

    ablations = results["ablation"]["ablations"]
    if ablations:
        lines.append("## Ablacion")
        lines.append("")
        for size, ablation in ablations.items():
            lines.append(
                f"- Top {size}: drop bal_acc={ablation['balanced_accuracy_drop']:.4f}, "
                f"drop f1_macro={ablation['f1_macro_drop']:.4f}, "
                f"drop aleatorio bal_acc={ablation['balanced_accuracy_drop_random']:.4f}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _truncate_text(text: str, max_chars: int = 280) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _top_words(texts: Sequence[str], top_n: int = 12) -> List[Dict[str, object]]:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = re.findall(r"[A-Za-z][A-Za-z']{2,}", text.lower())
        for token in tokens:
            if token in STOPWORDS:
                continue
            counter[token] += 1
    return [
        {"token": token, "count": int(count)}
        for token, count in counter.most_common(top_n)
    ]


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()