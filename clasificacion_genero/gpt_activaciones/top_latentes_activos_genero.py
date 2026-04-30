"""Calcula SOLO los latentes SAE mas activos por clase (genero) en TRAIN.

No entrena clasificador, no hace ablacion, no usa coeficientes.
Solo agrega la activacion media (y nonzero rate) por clase para todos los
latentes y devuelve el top-K por mayor activacion media en cada genero.
"""

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from interpretabilidad_sae_common import (  # type: ignore
    SaeInterpretabilityConfig,
    TOP_EXAMPLES_PER_LATENT,
    _aggregate_user_features,
    _collect_examples_for_target_latents,
    _compute_top_activation_per_class,
    _load_dataframe,
    _load_user_splits,
    _setup_models,
)

TOP_K = 30
TOP_EXAMPLES = TOP_EXAMPLES_PER_LATENT


def main() -> None:
    config = SaeInterpretabilityConfig(
        task_name="genero",
        dataset_kind="gender",
        cache_dir="/hdd/aitziber.l/activaciones_sae_gpt2_genero",
        splits_dir="data/splits_genero_70_15_15",
        output_dir="modelos/genero_sae_interpretabilidad",
        class_names=("female", "male"),
        top_k_latents=TOP_K,
        top_examples_per_latent=TOP_EXAMPLES,
    )

    print("TOP LATENTES SAE MAS ACTIVOS POR CLASE - genero", flush=True)

    df = _load_dataframe(config)
    train_auth, _eval_auth, _test_auth = _load_user_splits(config)

    df["author"] = df["author"].astype(str).str.strip()
    df_train = df[df["author"].isin(set(train_auth))].reset_index(drop=True)
    print(f"Train comments: {len(df_train):,}", flush=True)

    tokenizer, model, sae, hookpoint_module, num_latents = _setup_models()
    print(f"num_latents={num_latents}", flush=True)

    _train_users, class_act_stats = _aggregate_user_features(
        df=df_train,
        tokenizer=tokenizer,
        model=model,
        sae=sae,
        hookpoint_module=hookpoint_module,
        num_latents=num_latents,
        pass_name="TRAIN",
        pooling="mean_of_mean",
        num_classes=len(config.class_names),
    )

    # Primero calculamos el ranking sin ejemplos para saber que latentes nos
    # interesan; despues hacemos un segundo pase sobre TRAIN para recolectar
    # los textos con mayor activacion y derivar palabras asociadas.
    top_by_act_no_examples = _compute_top_activation_per_class(
        class_act_stats,
        config.class_names,
        top_k=TOP_K,
    )
    activation_targets = {}
    for class_idx, class_name in enumerate(config.class_names):
        latents = [int(e["latent_id"]) for e in top_by_act_no_examples.get(class_name, [])]
        if latents:
            activation_targets[class_idx] = latents

    if activation_targets:
        activation_examples = _collect_examples_for_target_latents(
            df=df_train,
            tokenizer=tokenizer,
            model=model,
            sae=sae,
            hookpoint_module=hookpoint_module,
            num_latents=num_latents,
            targets=activation_targets,
            top_examples=TOP_EXAMPLES,
            pooling="mean_of_mean",
            pass_name="TRAIN_TOPACT_EXAMPLES",
        )
    else:
        activation_examples = None

    del model, tokenizer, sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    top_by_act = _compute_top_activation_per_class(
        class_act_stats,
        config.class_names,
        top_k=TOP_K,
        examples_by_class_idx=activation_examples,
    )

    # Imprimir resultados al log
    print("\n" + "=" * 70, flush=True)
    print(f"TOP {TOP_K} latentes mas activos por clase (independiente del clasificador)", flush=True)
    for class_name, entries in top_by_act.items():
        print(f"\n[Clase {class_name}] top {len(entries)} latentes por activacion media:", flush=True)
        print(
            f"  {'rank':>4} {'latent':>7} {'mean_act':>12} {'nonzero':>10} "
            f"{'mean_other':>12} {'contrast':>12}  palabras",
            flush=True,
        )
        for rank, entry in enumerate(entries, 1):
            words = ", ".join(w["token"] for w in entry.get("top_words", [])[:8])
            print(
                f"  {rank:>4} L{entry['latent_id']:>6} "
                f"{entry['mean_activation']:>12.6f} "
                f"{entry['nonzero_rate']:>10.4f} "
                f"{entry['mean_activation_other_classes']:>12.6f} "
                f"{entry['mean_contrast']:>+12.6f}  [{words}]",
                flush=True,
            )

    os.makedirs(config.output_dir, exist_ok=True)
    out_path = os.path.join(config.output_dir, "top_latentes_activos_por_clase.json")
    payload = {
        "task_name": config.task_name,
        "class_names": list(config.class_names),
        "num_comments_train": int(len(df_train)),
        "num_latents": int(num_latents),
        "pooling": "mean_of_mean",
        "top_k": TOP_K,
        "top_examples_per_latent": TOP_EXAMPLES,
        "top_latents_by_activation": top_by_act,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {out_path}", flush=True)


if __name__ == "__main__":
    main()
