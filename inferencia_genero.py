"""
Inferencia de género usando SAE + Clasificador entrenado.

Este script permite predecir el género de nuevos textos usando:
1. SAE entrenada para extraer características latentes
2. Clasificador supervisado entrenado sobre esas características
"""

import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
import torch
from tiny_sae import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer


# =====================
# CONFIGURACIÓN
# =====================

MODEL = "openai-community/gpt2"
CONTEXT_LEN = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

PATH_SAE = "sae-ckpts/sae-gpt2-genero"
PATH_CLASIFICADOR = "modelos/clasificador_genero.pkl"


# =====================
# CLASE PREDICTOR
# =====================


class PredictorGenero:
    """Predictor de género que usa SAE + clasificador."""

    def __init__(
        self,
        path_sae: str = PATH_SAE,
        path_clasificador: str = PATH_CLASIFICADOR,
        model_name: str = MODEL,
        device: str = DEVICE,
    ):
        print("Inicializando PredictorGenero...")

        # Cargar SAE
        if not Path(path_sae).exists():
            raise FileNotFoundError(f"No se encuentra la SAE en {path_sae}")
        print(f"Cargando SAE desde {path_sae}...")
        self.sae = Sae.load_from_disk(path_sae, device=device)
        print(f"SAE cargada: {self.sae.cfg.num_latents} latentes, k={self.sae.cfg.k}")

        # Cargar clasificador
        if not Path(path_clasificador).exists():
            raise FileNotFoundError(
                f"No se encuentra el clasificador en {path_clasificador}"
            )
        print(f"Cargando clasificador desde {path_clasificador}...")
        self.clf = joblib.load(path_clasificador)

        # Cargar modelo y tokenizer
        print(f"Cargando modelo {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()

        self.device = device
        self.hookpoint = self.model.get_submodule(self.sae.cfg.hookpoint)

        print("✓ PredictorGenero listo para usar.\n")

    def predecir(self, textos: List[str]) -> dict:
        """Predice el género para una lista de textos.

        Parameters
        ----------
        textos : List[str]
            Lista de textos a clasificar.

        Returns
        -------
        dict con:
            - 'predicciones': lista de 'male' o 'female'
            - 'probabilidades': probabilidades [P(female), P(male)]
            - 'features': características extraídas (opcional, para análisis)
        """

        # Tokenizar
        tokens = self.tokenizer(
            textos,
            max_length=CONTEXT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        # Extraer activaciones intermedias
        captured_activations = None

        def hook(module, inputs, outputs):
            nonlocal captured_activations
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            captured_activations = outputs.detach()

        handle = self.hookpoint.register_forward_hook(hook)

        try:
            with torch.no_grad():
                # Forward pass
                _ = self.model(**tokens)

                # Activaciones capturadas
                acts = captured_activations.to(self.sae.dtype).to(self.sae.device)

                # Codificar con SAE
                top_acts, top_indices = self.sae.encode(acts)

                # Convertir a representación densa
                batch_size, seq_len, k = top_acts.shape
                dense_acts = torch.zeros(
                    batch_size, seq_len, self.sae.cfg.num_latents, device=self.sae.device
                )
                for b in range(batch_size):
                    for s in range(seq_len):
                        dense_acts[b, s, top_indices[b, s]] = top_acts[b, s]

                # Mean pooling
                features = dense_acts.mean(dim=1).cpu().numpy()  # (batch, num_latents)

        finally:
            handle.remove()

        # Clasificar
        predicciones_num = self.clf.predict(features)
        probabilidades = self.clf.predict_proba(features)

        # Convertir a labels
        predicciones = ["female" if p == 0 else "male" for p in predicciones_num]

        return {
            "predicciones": predicciones,
            "probabilidades": probabilidades,
            "features": features,
        }

    def predecir_uno(self, texto: str) -> dict:
        """Predice el género para un único texto."""
        resultado = self.predecir([texto])
        return {
            "prediccion": resultado["predicciones"][0],
            "probabilidad_female": resultado["probabilidades"][0][0],
            "probabilidad_male": resultado["probabilidades"][0][1],
            "features": resultado["features"][0],
        }


# =====================
# EJEMPLO DE USO
# =====================


def main():
    print("=" * 60)
    print("INFERENCIA DE GÉNERO CON SAE")
    print("=" * 60)
    print()

    # Inicializar predictor
    predictor = PredictorGenero()

    # Ejemplos de textos
    textos_ejemplo = [
        "I love shopping and makeup! Going to the mall is my favorite activity.",
        "Just finished fixing my car engine. Nothing beats working with your hands.",
        "I'm studying computer science and I love artificial intelligence research.",
        "My favorite hobby is knitting and baking cookies for my family.",
        "I enjoy playing video games and watching sports on weekends.",
    ]

    print("Prediciendo género para textos de ejemplo...\n")

    for i, texto in enumerate(textos_ejemplo, 1):
        resultado = predictor.predecir_uno(texto)
        print(f"Texto {i}: {texto[:80]}...")
        print(f"  → Predicción: {resultado['prediccion']}")
        print(
            f"  → Probabilidades: female={resultado['probabilidad_female']:.3f}, "
            f"male={resultado['probabilidad_male']:.3f}"
        )
        print()

    print("=" * 60)
    print("Para usar con tus propios textos:")
    print("  predictor = PredictorGenero()")
    print("  resultado = predictor.predecir_uno('tu texto aquí')")
    print("=" * 60)


if __name__ == "__main__":
    main()
