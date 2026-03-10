import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


def cargar_comentarios(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
	"""Carga el CSV de comentarios.

	Parameters
	----------
	path: str
		Ruta al CSV de comentarios.
	nrows: Optional[int]
		Número de filas a leer (útil para hacer pruebas con muestras grandes).
	"""

	path_obj = Path(path)
	if not path_obj.exists():
		raise FileNotFoundError(f"No se ha encontrado el fichero de comentarios: {path}")

	df = pd.read_csv(path_obj, nrows=nrows)

	# Normalizamos nombre de columna de autor si fuera necesario
	posibles_autor = ["author", "usuario", "user", "comment_author"]
	if "author" not in df.columns:
		for col in posibles_autor:
			if col in df.columns:
				df = df.rename(columns={col: "author"})
				break

	if "author" not in df.columns:
		raise ValueError("El CSV de comentarios debe contener una columna 'author' (o equivalente).")

	# Nos aseguramos de que el identificador de autor es string y sin espacios raros
	df["author"] = df["author"].astype(str).str.strip()

	return df


def cargar_autores(path: str) -> pd.DataFrame:
	"""Carga el CSV de autores con información de género y otros rasgos."""

	path_obj = Path(path)
	if not path_obj.exists():
		raise FileNotFoundError(f"No se ha encontrado el fichero de autores: {path}")

	df = pd.read_csv(path_obj)

	if "author" not in df.columns:
		raise ValueError("El CSV de autores debe contener una columna 'author'.")

	df["author"] = df["author"].astype(str).str.strip()

	return df


def normalizar_genero(df_autores: pd.DataFrame) -> pd.DataFrame:
	"""Crea una columna 'gender_clean' más consistente a partir de 'gender' e 'is_female'.

	- gender_clean = 'f' si is_female == 1 o gender == 'f'
	- gender_clean = 'm' si is_female == 0 o gender == 'm'
	- 'unknown' en el resto de casos
	"""

	df = df_autores.copy()

	# Normalizamos texto de gender si existe
	if "gender" in df.columns:
		df["gender"] = df["gender"].astype(str).str.lower().str.strip()
	else:
		df["gender"] = pd.NA

	# Creamos columna base con NaN
	df["gender_clean"] = pd.NA

	# Información de is_female si existe
	if "is_female" in df.columns:
		df.loc[df["is_female"] == 1, "gender_clean"] = "f"
		df.loc[df["is_female"] == 0, "gender_clean"] = "m"

	# Completamos con la variable gender original
	df.loc[df["gender_clean"].isna() & (df["gender"] == "f"), "gender_clean"] = "f"
	df.loc[df["gender_clean"].isna() & (df["gender"] == "m"), "gender_clean"] = "m"

	# Resto como unknown
	df["gender_clean"] = df["gender_clean"].fillna("unknown")

	return df


def calcular_num_comentarios_por_autor(df_comentarios: pd.DataFrame) -> pd.DataFrame:
	"""Devuelve un DataFrame con el número de comentarios por autor.

	Columns
	-------
	author: identificador del autor
	num_comments: número de comentarios escritos por ese autor
	"""

	if "author" not in df_comentarios.columns:
		raise ValueError("El DataFrame de comentarios debe tener una columna 'author'.")

	conteo = (
		df_comentarios
		.groupby("author", as_index=False)
		.size()
		.rename(columns={"size": "num_comments"})
	)
	return conteo


def unir_comentarios_y_autores(
	df_conteo_comentarios: pd.DataFrame, df_autores: pd.DataFrame
) -> pd.DataFrame:
	"""Une las tablas de conteo de comentarios y autores por 'author'."""

	columnas_necesarias = {"author", "num_comments"}
	if not columnas_necesarias.issubset(df_conteo_comentarios.columns):
		raise ValueError(
			"df_conteo_comentarios debe contener las columnas 'author' y 'num_comments'."
		)

	if "author" not in df_autores.columns:
		raise ValueError("df_autores debe contener la columna 'author'.")

	df_merged = df_conteo_comentarios.merge(df_autores, on="author", how="left")
	return df_merged


def _moda_serie(serie: pd.Series) -> Optional[float]:
	"""Devuelve la primera moda de una serie numérica o None si no existe."""

	if serie.empty:
		return None
	try:
		modos = serie.mode(dropna=True)
		if len(modos) == 0:
			return None
		return float(modos.iloc[0])
	except Exception:
		return None


def analisis_exploratorio_por_genero(df_merged: pd.DataFrame) -> Dict[str, Any]:
	"""Realiza un análisis exploratorio centrado en género.

	Espera un DataFrame con al menos las columnas:
	- 'num_comments': nº de comentarios por autor
	- 'gender_clean': género normalizado ('m', 'f', 'unknown')

	Devuelve un diccionario con resúmenes estadísticos y también imprime
	tablas legibles por pantalla.
	"""

	if "num_comments" not in df_merged.columns:
		raise ValueError("df_merged debe tener la columna 'num_comments'.")
	if "gender_clean" not in df_merged.columns:
		raise ValueError("df_merged debe tener la columna 'gender_clean'. Usa normalizar_genero primero.")

	df = df_merged.copy()

	# Aseguramos tipo numérico de num_comments
	df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce")

	# Resumen de autores por género
	autores_por_genero = df.groupby("gender_clean")["author"].nunique().rename("num_authors")

	# Estadísticos descriptivos de num_comments por género
	desc_por_genero = (
		df.groupby("gender_clean")["num_comments"]
		.agg(["count", "mean", "median", "std", "min", "max"])
	)

	# Moda de num_comments por género
	moda_por_genero = (
		df.groupby("gender_clean")["num_comments"]
		.apply(_moda_serie)
		.rename("mode")
	)

	# Sumamos nº total de comentarios por género (suma de num_comments)
	total_comentarios_por_genero = (
		df.groupby("gender_clean")["num_comments"].sum().rename("total_comments")
	)

	# Unimos todo en una única tabla
	resumen_genero = (
		desc_por_genero
		.join(moda_por_genero)
		.join(autores_por_genero)
		.join(total_comentarios_por_genero)
		.reset_index()
	)

	print("\n===== RESUMEN POR GÉNERO (a nivel de autor) =====")
	print(resumen_genero)

	# También podemos sacar estadísticas globales (sin agrupar) de num_comments
	print("\n===== ESTADÍSTICAS GLOBALES DE num_comments =====")
	print(df["num_comments"].describe())

	resultados: Dict[str, Any] = {
		"resumen_por_genero": resumen_genero,
		"autores_por_genero": autores_por_genero,
		"total_comentarios_por_genero": total_comentarios_por_genero,
		"estadisticas_globales_num_comments": df["num_comments"].describe(),
	}

	return resultados


if __name__ == "__main__":
	# EJEMPLO DE USO: ajusta estas rutas a las de tu Drive
	# Comentarios
	path_comentarios = "data/coments/all_comments_since_2015.csv"

	# Autores (ejemplo: donde tengas el CSV de autores que has mostrado)
	path_autores = "data/autores/autores.csv"

	# 1. Cargar datos (puedes quitar nrows para usar todo el dataset)
	df_comentarios = cargar_comentarios(path_comentarios, nrows=200000)
	df_autores_raw = cargar_autores(path_autores)

	# 2. Normalizar género en la tabla de autores
	df_autores = normalizar_genero(df_autores_raw)

	# 3. Calcular nº de comentarios por autor
	df_conteo = calcular_num_comentarios_por_autor(df_comentarios)

	# 4. Unir conteo con info de autores (incluyendo gender_clean)
	df_merged = unir_comentarios_y_autores(df_conteo, df_autores)

	# 5. Análisis exploratorio centrado en género
	_ = analisis_exploratorio_por_genero(df_merged)

