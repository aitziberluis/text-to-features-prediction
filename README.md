Uso de activaciones originales frente a representaciones SAE en la inferencia de características demográficas y personales a partir de comentarios textuales (proyecto TFM Aitziber Luis Villamil)
===

Desarrollo de métodos que no solo sean capaces de predecir características personales y demográficas a partir de texto, sino que también permitan interpretar las representaciones internas utilizadas para ello o que se pueda analizar porque se ha obtenido ese resultado (el uso de SAE)

Resumen
-------

Este repositorio implementa un flujo de trabajo para:

- limpiar y unir comentarios con perfiles de autor.
- entrenar un autoencoder sparse (SAE) sobre activaciones de modelos causales (en este caso hay código para entrenar modelo de QWEN y de GPT-2).
- extraer activaciones de capas intermedias de GPT-2 y Qwen.
- clasificar genero, edad y dimensiones MBTI (rasgos de personalidad de perceptivo, intuitivo, introvertido, racional).
- analizar la interpretabilidad de las representaciones y los latentes.

Estructura y carpetas
---------------------
- `preprocesamiento.py` - carga, limpia y normaliza datos.
- `tiny_sae.py` - definicion de la SAE y funciones de entrenamiento.
- `sae_gpt.py` - entrenamiento SAE sobre GPT-2.
- `sae_qwen.py` - entrenamiento SAE sobre Qwen.
- `find_best_layer.py` - encontrar la capa que mejores resultados puede dar para esta tarea (se ha usado sobre el modelo de QWEN).
- `interpretabilidad_sae_common.py` - analisis de latentes SAE.
- `interpretabilidad_sae_qwen_common.py` - analisis SAE/Qwen.
- `clasificacion_edad/` - experimentos de edad.
- `clasificacion_genero/` - experimentos de genero.
- `clasificacion_mbti/` - experimentos MBTI.
- `data/` - datos de entrada.
- `figuras/` - graficos generados.
- `logs/` - registros de ejecucion.
- `modelos/` - resultados de clasificadores.
- `sae-ckpts/` - checkpoints SAE.
- `wandb/` - artefactos de seguimiento opcional.

Datos esperados
---------------
(Datos personales y con dimensiones muy grandes por eso no estan subidos)
- `data/all_comments_since_2015.csv`
  - columnas: `author`, `body`
- `data/author_profiles.csv`
  - columnas: `author`, `gender`, `is_female`, `age`, `introverted`, `intuitive`, `thinking`, `perceiving`

`preprocesamiento.py` valida los datos, normaliza `author`, filtra cuerpos vacios y crea datasets de entrenamiento.

Flujos principales
------------------

1. Preparacion de datos
   - `preprocesamiento.py` carga comentarios y perfiles.
   - normaliza genero y edad.
   - une comentarios con etiquetas por autor.

2. Entrenamiento SAE
   - `sae_gpt.py` entrena la SAE usando GPT-2.
   - `sae_qwen.py` entrena la SAE usando Qwen.
   - `tiny_sae.py` define la arquitectura SAE y `train_sae`.

3. Extraccion de activaciones
   - `find_best_layer.py` extrae y evalua activaciones por capa.
   - permite elegir la mejor capa para genero, edad y MBTI (se ha usado para elegir la mejor capa del modelo de QWEN).

4. Clasificacion y evaluacion
   - `clasificacion_edad/`, `clasificacion_genero/`, `clasificacion_mbti/`.
   - los scripts entrenan clasificadores lineales sobre activaciones y representaciones SAE.
   - tambien generan analisis de interpretabilidad y ablaciones.

5. Interpretabilidad
   - `interpretabilidad_sae_common.py` agrupa resultados por usuario.
   - selecciona latentes relevantes y calcula metricas de comportamiento.
   - guarda resúmenes JSON y markdown.

Ejecucion recomendada
---------------------

1. validar datos

```sh
cd TFM
python3 preprocesamiento.py
```

2. entrenar SAE con GPT-2

```sh
cd TFM
python3 sae_gpt.py
```

3. entrenar SAE con Qwen

```sh
cd TFM
python3 sae_qwen.py
```

4. ejecutar barrido de capas

```sh
cd TFM
python3 find_best_layer.py
```

5. clasificacion de genero

```sh
cd TFM/clasificacion_genero/gpt_qwen_activaciones
python3 clasificador_genero_gpt_activaciones.py
```

6. interpretabilidad de genero

```sh
cd TFM/clasificacion_genero/gpt_qwen_activaciones
python3 interpretabilidad_genero_sae.py
```

7. clasificacion de edad

```sh
cd TFM/clasificacion_edad/gpt_qwen_activaciones
python3 clasificador_edad_qwen_sae_activaciones.py
```

8. clasificacion MBTI

```sh
cd TFM/clasificacion_mbti/introverted
python3 clasificador_introverted_sae_activaciones.py
```



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

Notas
-----
- proyecto orientado a GPU; training y extraccion de activaciones pueden ser costosos.
- los checkpoints y activaciones pueden ocupar mucho espacio en disco.
- los experimentos MBTI son binarias para cada dimension.
- `wandb/` es opcional y se usa solo si se habilita seguimiento con weights and biases.
