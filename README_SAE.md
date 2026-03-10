# Clasificación de Género usando Sparse Autoencoders (SAE)

Este proyecto implementa un sistema de clasificación de género basado en texto utilizando **Sparse Autoencoders (SAE)** para extraer representaciones interpretables de comentarios de Reddit, seguido de un clasificador supervisado.

## 📋 Descripción del Proyecto

El objetivo es **inferir el género de una persona a partir de sus textos** utilizando las representaciones aprendidas por una SAE. El flujo completo es:

1. **Preprocesamiento** → Análisis exploratorio de datos (EDA) centrado en género
2. **Entrenamiento de SAE** → Aprender representaciones sparse no supervisadas de los textos
3. **Clasificador supervisado** → Entrenar un modelo de clasificación usando las características SAE
4. **Inferencia** → Predecir género de nuevos textos

## 🗂️ Estructura de Archivos

```
TFM/
├── data/
│   ├── all_comments_since_2015.csv    # Comentarios de Reddit
│   └── author_profiles.csv            # Perfiles de autores con género
├── preprocesamiento.py                # EDA y análisis por género
├── sae_genero.py                      # Entrenamiento de la SAE
├── clasificador_genero.py             # Entrenamiento del clasificador
├── inferencia_genero.py               # Inferencia en nuevos textos
├── sae-ckpts/                         # Checkpoints de la SAE
│   └── sae-gpt2-genero/
└── modelos/                           # Clasificador entrenado
    └── clasificador_genero.pkl
```

## 🚀 Pipeline Completo

### Paso 1: Análisis Exploratorio de Datos

Ejecuta el preprocesamiento para ver estadísticas descriptivas por género:

```bash
python3 preprocesamiento.py
```

**Salida**: Resumen por género (media, mediana, moda de comentarios por autor), distribución de los datos, etc.

### Paso 2: Entrenar la SAE

Entrena una Sparse Autoencoder sobre las activaciones internas de GPT-2 usando tus comentarios:

```bash
python3 sae_genero.py
```

**¿Qué hace?**
- Carga comentarios y los asocia con género (m/f) por autor
- Tokeniza textos con GPT-2 (longitud 512)
- Entrena una SAE de `tiny_sae` en la capa `transformer.h.8`
- Guarda la SAE entrenada en `sae-ckpts/sae-gpt2-genero/`

**Configuración de la SAE:**
- `d_in`: 768 (dimensión de GPT-2 small)
- `num_latents`: 16,384 (2^14)
- `k`: 64 (activaciones sparse - top-k)
- `hookpoint`: `transformer.h.8`

**Nota**: Este entrenamiento es **no supervisado**. La SAE aprende a reconstruir activaciones sin usar el género.

### Paso 3: Entrenar el Clasificador de Género

Usa la SAE entrenada como extractor de características y entrena un clasificador supervisado:

```bash
python3 clasificador_genero.py
```

**¿Qué hace?**
- Carga la SAE entrenada
- Extrae características latentes (códigos sparse) para cada comentario
- Agrega las activaciones por comentario (mean pooling sobre tokens)
- Entrena una **Regresión Logística** con esas características
- Evalúa en test set (80/20 split)
- Guarda el clasificador en `modelos/clasificador_genero.pkl`

**Salida esperada:**
```
Accuracy: 0.XXXX
F1-score (weighted): 0.XXXX

Classification Report:
              precision    recall  f1-score   support
      female       0.XX      0.XX      0.XX      XXXX
        male       0.XX      0.XX      0.XX      XXXX
```

### Paso 4: Inferencia en Nuevos Textos

Predice el género de nuevos textos:

```bash
python3 inferencia_genero.py
```

**Uso programático:**

```python
from inferencia_genero import PredictorGenero

# Inicializar predictor
predictor = PredictorGenero()

# Predecir un texto
resultado = predictor.predecir_uno("I love programming and building AI models.")

print(resultado['prediccion'])           # 'male' o 'female'
print(resultado['probabilidad_male'])    # ej: 0.723
print(resultado['probabilidad_female'])  # ej: 0.277

# Predecir múltiples textos
textos = ["texto 1", "texto 2", "texto 3"]
resultados = predictor.predecir(textos)
```

## 📊 Detalles Técnicos

### Sparse Autoencoder (SAE)

- **Arquitectura**: Top-k sparse autoencoder basado en [tiny_sae](https://github.com/ApolloResearch/tiny_sae)
- **Sparsity**: Solo las k=64 activaciones más altas se mantienen (de 16,384 latentes)
- **Loss**: FVU (Fraction of Variance Unexplained) - reconstrucción de activaciones
- **Normalización**: Decodificador con norma unitaria en cada paso

### Clasificador

- **Modelo**: Logistic Regression con `class_weight='balanced'`
- **Features**: Representaciones sparse agregadas (mean pooling sobre secuencia)
- **Dimensionalidad**: 16,384 características (num_latents de la SAE)
- **Split**: 80% train, 20% test, estratificado por género

### Preprocesamiento de Género

La función `normalizar_genero()` crea una columna `gender_clean` a partir de:
- `gender` (columna original: 'm', 'f', NaN)
- `is_female` (columna binaria: 0, 1, NaN)

**Lógica:**
- `gender_clean = 'f'` si `is_female == 1` o `gender == 'f'`
- `gender_clean = 'm'` si `is_female == 0` o `gender == 'm'`
- `gender_clean = 'unknown'` en otros casos

Solo se usan comentarios con `gender_clean` en `['m', 'f']` para el entrenamiento.

## 🛠️ Requisitos

```bash
pip install torch transformers datasets tiny-sae pandas scikit-learn tqdm wandb python-dotenv safetensors joblib
```

**Hardware recomendado:**
- GPU con ≥8GB VRAM (para entrenamiento de SAE)
- ≥16GB RAM

## 📈 Monitoreo con Weights & Biases

El entrenamiento de la SAE usa W&B para tracking:
- Proyecto: `tiny-sae-genero`
- Run: `sae-gpt2-genero`
- Métrica: FVU (Fraction of Variance Unexplained)

Configura tu API key de W&B en `.env`:
```
WANDB_API_KEY=tu_api_key_aqui
```

## 🔬 Análisis de Resultados

### Estadísticas por Género (ejemplo del EDA)

```
===== RESUMEN POR GÉNERO =====
  gender_clean  count   mean  median    std  min    max    mode  num_authors
0            f   1331   1847    832   2543    1  27451   317.0         1331
1            m   1753   2806   1255   3891    1  101785    1.0         1753
2      unknown   7211   1422    481   3102    1  89432    1.0         7211
```

- Los hombres (`m`) tienden a escribir más comentarios en promedio que las mujeres (`f`)
- Alta variabilidad dentro de cada grupo (std alta)
- La moda en hombres es 1 comentario (muchos usuarios ocasionales)

### Interpretabilidad de la SAE

Las SAEs aprenden características **interpretables**. Puedes analizar qué latentes se activan para textos con ciertos temas (por ejemplo, "deportes", "maquillaje", "tecnología") y ver si correlacionan con el género predicho.

## 🎯 Próximos Pasos

1. **Ajustar hiperparámetros**: `num_latents`, `k`, capa del hookpoint
2. **Probar otros clasificadores**: Random Forest, XGBoost, pequeña red neuronal
3. **Análisis de features importantes**: ¿Qué latentes correlacionan más con género?
4. **Cross-validation**: Para estimación más robusta del rendimiento
5. **Análisis de sesgo**: Evaluar si el modelo perpetúa estereotipos de género

## 📚 Referencias

- [Sparse Autoencoders for Interpretability](https://arxiv.org/abs/2309.08600)
- [tiny_sae library](https://github.com/ApolloResearch/tiny_sae)
- [GPT-2 by OpenAI](https://openai.com/research/better-language-models)

## 📝 Notas

- El dataset original tiene ~17M comentarios; puedes limitar con `MAX_COMMENTS` en cada script para pruebas rápidas
- El entrenamiento de la SAE puede tardar varias horas dependiendo del tamaño del dataset y GPU
- Los checkpoints se guardan cada 10M tokens por defecto

---

**Autor**: Proyecto TFM - Clasificación de Género con SAE  
**Fecha**: Marzo 2026
