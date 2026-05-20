TFM
===

Proyecto de clasificacion e interpretabilidad de modelos de texto.

Objetivo
--------

Extraer representaciones de lenguaje natural de un corpus de comentarios y usar esas representaciones para:

- entrenar un autoencoder sparse (SAE) sobre activaciones de modelos causales.
- clasificar genero y edad del autor.
- clasificar dimensiones MBTI (introverted, intuitive, thinking, perceiving).
- evaluar interpretabilidad de las activaciones y la calidad de las capas intermedias.

Estructura principal
--------------------

- `preprocesamiento.py` - carga y normaliza datos de comentarios y perfiles de autor.
- `tiny_sae.py` - implementacion del autoencoder sparse y funciones de entrenamiento.
- `sae_gpt.py` - entrenamiento SAE sobre GPT-2 y preparacion de tokenizacion.
- `sae_qwen.py` - entrenamiento SAE sobre Qwen y mejoras de rendimiento CUDA.
- `find_best_layer.py` - barrido de capas para seleccionar la mejor capa de activacion.
- `interpretabilidad_sae_common.py` - funciones compartidas para análisis interpretativo.
- `interpretabilidad_sae_qwen_common.py` - funciones de interpretabilidad específicas para Qwen.
- `clasificacion_edad/` - scripts para clasificar rangos de edad con distintas fuentes de activaciones.
- `clasificacion_genero/` - scripts para clasificar genero.
- `clasificacion_mbti/` - scripts para clasificar dimensiones MBTI.
- `data/` - datos de entrada: comentarios y perfiles de autor.
- `figuras/` - gráficos generados.
- `logs/` - registros de entrenamiento y evaluacion.
- `sae-ckpts/` - checkpoints de SAE entrenados.
- `modelos/` - resultados de modelos y salidas de clasificadores.
- `wandb/` - datos de Weights & Biases si se usa seguimiento de experimentos.

Datos esperados
---------------

- `data/all_comments_since_2015.csv` - dataset de comentarios.
  - columna de texto esperada: `body`.
  - columna de autor esperada: `author` o equivalente.
- `data/author_profiles.csv` - perfiles de autor.
  - columna de autor: `author`.
  - columnas de genero: `gender`, `is_female`.
  - columna de edad: `age`.
  - columnas MBTI: `introverted`, `intuitive`, `thinking`, `perceiving`.

Tareas principales
------------------

1. preparacion de datos
   - carga comentarios limpios y perfiles de autor.
   - normaliza genero y edad.
   - une comentarios con etiquetas por autor.

2. entrenamiento SAE
   - `sae_gpt.py` usa GPT-2 y `tiny_sae.py` para entrenar un autoencoder sparse.
   - `sae_qwen.py` usa Qwen y optimizaciones de CUDA / flash attention.

3. extracción de activaciones
   - `find_best_layer.py` prueba capas intermedias de un modelo causal.
   - clasificacion basada en activaciones GPT/Qwen y en representaciones SAE.

4. clasificación y evaluacion
   - carpetas `clasificacion_edad`, `clasificacion_genero`, `clasificacion_mbti`.
   - se entrena y evalua clasificación sobre datos de usuario y datos de comentario.
   - se comparan configuraciones con y sin balanceo de clases.

Uso básico
-----------

1. colocar los CSV en `data/`.
2. revisar y ajustar rutas en los scripts o variables de entorno.
3. ejecutar `preprocesamiento.py` para verificar carga y limpiar columnas.
4. entrenar SAE con `sae_gpt.py` o `sae_qwen.py`.
5. ejecutar los experimentos de clasificación en las carpetas correspondientes.
6. revisar `logs/`, `figuras/` y `modelos/`.

Dependencias
------------

- python 3.10+
- pandas
- numpy
- scikit-learn
- torch
- transformers
- datasets
- matplotlib
- seaborn
- wandb
- scipy
- einops
- safetensors

Notas de entorno
----------------

- muchos scripts leen variables de entorno para control de batch size, device y rutas.
- se recomienda `cuda` cuando haya GPU disponible.
- los checkpoints pueden guardarse fuera del directorio del proyecto en discos rápidos.

Puntos de atención
------------------

- `preprocesamiento.py` es el punto central para unir comentarios y autor.
- `find_best_layer.py` ayuda a elegir la capa de activacion con mejor performance.
- los experimentos de MBTI son binarias por cada dimensión.
- `sae-ckpts/` y `logs/` contienen resultados que pueden ocupar espacio.
