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
import pandas as pd
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
    """Predictor de género que usa SAE + clasificador.
    
    IMPORTANTE: Este predictor clasifica en género BINARIO:
    - 'female' (f)
    - 'male' (m)
    
    No predice 'unknown' porque el modelo fue entrenado solo con 'f' y 'm'.
    """

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

    def _extraer_features(self, textos: List[str]) -> np.ndarray:
        """Extrae features SAE para una lista de textos."""

        if not textos:
            raise ValueError("La lista de textos no puede estar vacía.")

        # Limpiamos entradas vacías para evitar tokenizaciones inútiles.
        textos_limpios = [t for t in textos if isinstance(t, str) and t.strip()]
        if not textos_limpios:
            raise ValueError("No hay textos válidos tras limpiar entradas vacías.")

        # Tokenizar
        tokens = self.tokenizer(
            textos_limpios,
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

        return features

    def predecir(self, textos: List[str]) -> dict:
        """Predice el género para una lista de textos.

        Returns
        -------
        dict con:
            - 'predicciones': lista de 'male' o 'female'
            - 'probabilidades': np.ndarray con [P(female), P(male)] por texto
            - 'features': características SAE por texto
        """

        features = self._extraer_features(textos)

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

    def predecir_usuario(self, textos_usuario: List[str]) -> dict:
        """Predice género a nivel usuario agregando varios textos.

        La salida principal es una predicción única basada en la media de
        probabilidades por comentario.
        """

        resultado_textos = self.predecir(textos_usuario)
        probabilidades = resultado_textos["probabilidades"]

        prob_media = probabilidades.mean(axis=0)
        prob_std = probabilidades.std(axis=0)
        pred_num = int(np.argmax(prob_media))
        pred_label = "female" if pred_num == 0 else "male"

        return {
            "prediccion_usuario": pred_label,
            "probabilidad_media_female": float(prob_media[0]),
            "probabilidad_media_male": float(prob_media[1]),
            "incertidumbre_female": float(prob_std[0]),
            "incertidumbre_male": float(prob_std[1]),
            "num_textos": int(probabilidades.shape[0]),
            "detalle_por_texto": {
                "predicciones": resultado_textos["predicciones"],
                "probabilidades": probabilidades,
            },
        }

    def predecir_usuario_desde_csv(
        self,
        author: str,
        path_comentarios: str,
        text_column: str = "body",
        author_column: str = "author",
        max_textos: int = 200,
    ) -> dict:
        """Predice el género de un autor leyendo sus textos desde CSV."""

        if not Path(path_comentarios).exists():
            raise FileNotFoundError(f"No se encuentra el CSV: {path_comentarios}")

        df = pd.read_csv(path_comentarios)

        if author_column not in df.columns:
            raise ValueError(
                f"La columna de autor '{author_column}' no existe en {path_comentarios}"
            )
        if text_column not in df.columns:
            raise ValueError(
                f"La columna de texto '{text_column}' no existe en {path_comentarios}"
            )

        author_norm = str(author).strip()
        df_author = df[df[author_column].astype(str).str.strip() == author_norm]

        if df_author.empty:
            raise ValueError(f"No se encontraron textos para el autor '{author_norm}'.")

        textos = (
            df_author[text_column]
            .dropna()
            .astype(str)
            .str.strip()
        )
        textos = textos[textos != ""]

        if max_textos is not None and max_textos > 0:
            textos = textos.iloc[:max_textos]

        if len(textos) == 0:
            raise ValueError(f"El autor '{author_norm}' no tiene textos válidos.")

        salida = self.predecir_usuario(textos.tolist())
        salida["author"] = author_norm
        return salida

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

    print("Prediciendo género para textos de ejemplo (por texto)...\n")

    for i, texto in enumerate(textos_ejemplo, 1):
        resultado = predictor.predecir_uno(texto)
        print(f"Texto {i}: {texto[:80]}...")
        print(f"  → Predicción: {resultado['prediccion']}")
        print(
            f"  → Probabilidades: female={resultado['probabilidad_female']:.3f}, "
            f"male={resultado['probabilidad_male']:.3f}"
        )
        print()

    print("Predicción agregada a nivel usuario (con los 5 textos):")
    res_usuario = predictor.predecir_usuario(textos_ejemplo)
    print(f"  → Predicción usuario: {res_usuario['prediccion_usuario']}")
    print(
        "  → Probabilidades medias: "
        f"female={res_usuario['probabilidad_media_female']:.3f}, "
        f"male={res_usuario['probabilidad_media_male']:.3f}"
    )
    print(
        "  → Incertidumbre (desv. estándar entre textos): "
        f"female={res_usuario['incertidumbre_female']:.3f}, "
        f"male={res_usuario['incertidumbre_male']:.3f}"
    )
    print()

    print("=" * 60)
    print("Para usar con tus propios textos:")
    print("  predictor = PredictorGenero()")
    print("  resultado = predictor.predecir_uno('tu texto aquí')")
    print("  resultado_usuario = predictor.predecir_usuario(lista_textos_usuario)")
    print("  resultado_csv = predictor.predecir_usuario_desde_csv(author='usuario', path_comentarios='data/all_comments_since_2015.csv')")
    print("=" * 60)


if __name__ == "__main__":
    main()
