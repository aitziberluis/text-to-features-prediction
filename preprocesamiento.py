import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def cargar_comentarios(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
	"""Carga el CSV de comentarios.
	"""
	path_obj = Path(path)
	if not path_obj.exists():
		raise FileNotFoundError(f"No se ha encontrado el fichero de comentarios: {path}")
	df = pd.read_csv(path_obj, nrows=nrows)
	#Normalizamos nombre de columna de autor si fuera necesario
	posibles_autor = ["author", "usuario", "user", "comment_author"]
	if "author" not in df.columns:
		for col in posibles_autor:
			if col in df.columns:
				df = df.rename(columns={col: "author"})
				break
	if "author" not in df.columns:
		raise ValueError("El CSV de comentarios debe contener una columna 'author' (o equivalente).")
	#Nos aseguramos de que el identificador de autor es string y sin espacios raros
	df["author"] = df["author"].astype(str).str.strip()
	# Limpiamos filas con cuerpo de comentario nulo o vacío.
	if "body" in df.columns:
		antes = len(df)
		df["body"] = df["body"].astype("string").str.strip()
		df = df[df["body"].notna() & (df["body"] != "")].reset_index(drop=True)
		eliminadas = antes - len(df)
		if eliminadas > 0:
			print(f"Se eliminaron {eliminadas:,} filas de comentarios con 'body' nulo o vacío.")
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
	- gender_clean = 'f' si is_female == 1 o gender == 'f'- gender_clean = 'm' si is_female == 0 o gender == 'm'- 'unknown' en el resto de casos"""
	df = df_autores.copy()
	#Normalizamos texto de gender si existe
	if "gender" in df.columns:
		df["gender"] = df["gender"].astype(str).str.lower().str.strip()
	else:
		df["gender"] = pd.NA
	#Creamos columna base con NaN
	df["gender_clean"] = pd.NA
	#Información de is_female si existe
	if "is_female" in df.columns:
		df.loc[df["is_female"] == 1, "gender_clean"] = "f"
		df.loc[df["is_female"] == 0, "gender_clean"] = "m"
	#Completamos con la variable gender original
	df.loc[df["gender_clean"].isna() & (df["gender"] == "f"), "gender_clean"] = "f"
	df.loc[df["gender_clean"].isna() & (df["gender"] == "m"), "gender_clean"] = "m"
	#Resto como unknown
	df["gender_clean"] = df["gender_clean"].fillna("unknown")

	return df


DEFAULT_AGE_BINS = [14, 20, 30, 40, np.inf]
DEFAULT_AGE_GROUP_LABELS = ["14_19", "20_29", "30_39", "40_plus"]

def normalizar_edad(
	df_autores: pd.DataFrame,
	age_bins: Optional[list] = None,
	age_labels: Optional[list] = None,
) -> pd.DataFrame:
	"""Crea columnas limpias de edad y rango de edad a partir de `age`.
	Por defecto usa rangos adaptados a la distribución real del dataset:
	- 14_19
	- 20_29
	- 30_39
	- 40_plus
	"""
	if age_bins is None:
		age_bins = DEFAULT_AGE_BINS
	if age_labels is None:
		age_labels = DEFAULT_AGE_GROUP_LABELS
	if len(age_bins) != len(age_labels) + 1:
		raise ValueError("age_bins debe tener exactamente un elemento más que age_labels.")
	df = df_autores.copy()
	if "age" not in df.columns:
		raise ValueError("El CSV de autores debe contener una columna 'age' para preparar el dataset de edad.")
	df["age_clean"] = pd.to_numeric(df["age"], errors="coerce")
	df.loc[df["age_clean"] <= 0, "age_clean"] = pd.NA
	df["age_group"] = pd.cut(df["age_clean"],bins=age_bins,labels=age_labels,right=False,include_lowest=True,
	)
	df["age_group"] = df["age_group"].astype("string")
	df["age_group"] = df["age_group"].fillna("unknown")
	return df

def _calcular_entropia_columna(serie: pd.Series) -> float:
	"""Calcula la entropía de Shannon de una columna en base 2."""
	value_probs = serie.astype(str).value_counts(normalize=True, dropna=False)
	if value_probs.empty:
		return 0.0
	return float(-(value_probs * np.log2(value_probs + 1e-12)).sum())

def _mutual_info_discretizada(
	serie_x: pd.Series,
	serie_y: pd.Series,
	n_bins: int = 10,
) -> float:
	"""Calcula información mutua entre dos variables numéricas discretizando por cuantiles."""
	x = pd.to_numeric(serie_x, errors="coerce")
	y = pd.to_numeric(serie_y, errors="coerce")
	mask = x.notna() & y.notna()
	x = x[mask]
	y = y[mask]
	if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
		return 0.0
	try:
		x_disc = pd.qcut(x, q=min(n_bins, x.nunique()), labels=False, duplicates="drop")
		y_disc = pd.qcut(y, q=min(n_bins, y.nunique()), labels=False, duplicates="drop")
	except Exception:
		return 0.0
	cont = pd.crosstab(x_disc, y_disc)
	if cont.empty:
		return 0.0
	pxy = cont.to_numpy(dtype=float)
	pxy = pxy / pxy.sum()
	px = pxy.sum(axis=1, keepdims=True)
	py = pxy.sum(axis=0, keepdims=True)
	nz = pxy > 0
	mi = np.sum(pxy[nz] * np.log2(pxy[nz] / (px @ py)[nz]))
	return float(max(mi, 0.0))

def _matriz_informacion_mutua(df_num: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
	"""Construye una matriz simétrica de información mutua para columnas numéricas."""
	cols = list(df_num.columns)
	mi_mat = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
	for i, c1 in enumerate(cols):
		for j in range(i, len(cols)):
			c2 = cols[j]
			if i == j:
				mi_val = _calcular_entropia_columna(pd.to_numeric(df_num[c1], errors="coerce").dropna())
			else:
				mi_val = _mutual_info_discretizada(df_num[c1], df_num[c2], n_bins=n_bins)
			mi_mat.iloc[i, j] = mi_val
			mi_mat.iloc[j, i] = mi_val
	return mi_mat

def _plot_densidades_numericas(
	df: pd.DataFrame,
	nombre_dataset: str,
	output_dir: str,
	numeric_cols: pd.Index,
	columnas_excluir: Optional[list] = None,
) -> Optional[str]:
	"""Genera funciones de densidad (KDE) para todas las columnas numéricas."""
	if columnas_excluir is None:
		columnas_excluir = []
	numeric_cols = pd.Index([c for c in numeric_cols if c not in set(columnas_excluir)])
	if len(numeric_cols) == 0:
		return None
	n_cols = min(3, len(numeric_cols))
	n_rows = int(np.ceil(len(numeric_cols) / n_cols))
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
	axes = np.array(axes).reshape(-1)
	for i, col in enumerate(numeric_cols):
		ax = axes[i]
		serie = pd.to_numeric(df[col], errors="coerce").dropna()
		# Si hay muy poca variación, usamos histograma para evitar errores de KDE.
		if len(serie) < 2 or serie.nunique() < 2:
			sns.histplot(serie, bins=20, kde=False, ax=ax, color="#4C72B0")
			ax.set_title(f"{col} (sin KDE por baja variación)", fontsize=10)
		else:
			sns.kdeplot(serie, fill=True, ax=ax, color="#4C72B0")
			ax.set_title(f"Densidad de {col}", fontsize=10)
		ax.set_xlabel(col)
		ax.set_ylabel("Densidad")
	for j in range(len(numeric_cols), len(axes)):
		axes[j].axis("off")
	plt.tight_layout()
	path_fig = os.path.join(output_dir, f"densidades_{nombre_dataset}.png")
	plt.savefig(path_fig, dpi=300, bbox_inches="tight")
	plt.close()
	return path_fig
def analizar_dataset_general(
	df: pd.DataFrame,
	nombre_dataset: str,
	output_dir: str = "figuras",
	columnas_excluir_densidad: Optional[list] = None,
	columnas_excluir_correlacion: Optional[list] = None,
) -> Dict[str, Any]:
	"""Analiza calidad y estructura del dataset antes del preprocesamiento específico."""
	os.makedirs(output_dir, exist_ok=True)
	print(f"Análisis General Inicial - {nombre_dataset.upper()}")
	print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")
	#1) Valores nulos por columna
	nulos_abs = df.isna().sum().sort_values(ascending=False)
	nulos_pct = (100 * nulos_abs / len(df)).round(3) if len(df) > 0 else nulos_abs.astype(float)
	nulos_df = pd.DataFrame({"nulos": nulos_abs, "pct_nulos": nulos_pct})
	print("\n[1] Valores nulos por columna")
	print(nulos_df.to_string())
	#2) Duplicados
	dup_filas = int(df.duplicated().sum())
	dup_pct = (100 * dup_filas / len(df)) if len(df) > 0 else 0.0
	print("\n[2] Duplicados")
	print(f"Filas duplicadas exactas: {dup_filas:,} ({dup_pct:.3f}%)")
	#3) Tipos de datos
	dtypes_df = pd.DataFrame({
		"columna": df.columns,
		"dtype": df.dtypes.astype(str).values,
		"n_unicos": [df[c].nunique(dropna=True) for c in df.columns],
	})
	print("\n[3] Tipos de datos y cardinalidad")
	print(dtypes_df.to_string(index=False))
	#4) Entropía por columna
	entropias = pd.Series({col: _calcular_entropia_columna(df[col]) for col in df.columns}).sort_values(ascending=False)
	print("\n[4] Entropía por columna (Shannon, base 2)")
	print(entropias.round(4).to_string())
	col_max_entropia = entropias.index[0] if len(entropias) > 0 else None
	#5) Rangos por columna numérica
	numeric_cols = df.select_dtypes(include=[np.number]).columns
	rangos_df = pd.DataFrame(columns=["columna", "min", "max", "rango"])
	col_mayor_rango = None
	col_menor_rango = None
	if len(numeric_cols) > 0:
		rangos_df = pd.DataFrame({
			"columna": numeric_cols,
			"min": [pd.to_numeric(df[c], errors="coerce").min() for c in numeric_cols],
			"max": [pd.to_numeric(df[c], errors="coerce").max() for c in numeric_cols],
		})
		rangos_df["rango"] = rangos_df["max"] - rangos_df["min"]
		rangos_df = rangos_df.sort_values("rango", ascending=False).reset_index(drop=True)
		col_mayor_rango = rangos_df.iloc[0]["columna"] if len(rangos_df) > 0 else None
		col_menor_rango = rangos_df.iloc[-1]["columna"] if len(rangos_df) > 0 else None
		print("\n[5] Rango por columna numérica")
		print(rangos_df.to_string(index=False))
		print(f"Mayor rango: {col_mayor_rango}")
		print(f"Menor rango: {col_menor_rango}")
	else:
		print("\n[5] Rango por columna numérica")
		print("No hay columnas numéricas en este dataset.")
	#6) Correlaciones entre columnas numéricas
	correlaciones = pd.DataFrame()
	info_mutua = pd.DataFrame()
	path_corr = None
	path_mi = None
	numeric_cols_corr = numeric_cols
	if columnas_excluir_correlacion is not None:
		numeric_cols_corr = pd.Index([c for c in numeric_cols if c not in set(columnas_excluir_correlacion)])
	if len(numeric_cols_corr) >= 2:
		correlaciones = df[numeric_cols_corr].corr(numeric_only=True)
		print("\n[6] Matriz de correlación (Pearson)")
		print(correlaciones.round(4).to_string())
		fig, ax = plt.subplots(figsize=(1.2 * len(numeric_cols_corr) + 4, 1.0 * len(numeric_cols_corr) + 3))
		sns.heatmap(correlaciones, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
		ax.set_title(f"Heatmap de correlación - {nombre_dataset}", fontweight="bold")
		plt.tight_layout()
		path_corr = os.path.join(output_dir, f"correlacion_{nombre_dataset}.png")
		plt.savefig(path_corr, dpi=300, bbox_inches="tight")
		plt.close()
		print(f"Figura de correlación guardada: {path_corr}")
	else:
		print("\n[6] Matriz de correlación")
		print("Se necesitan al menos 2 columnas numéricas para calcular correlaciones.")
	#6b) Información mutua entre columnas numéricas
	if len(numeric_cols_corr) >= 2:
		info_mutua = _matriz_informacion_mutua(df[numeric_cols_corr], n_bins=10)
		print("\n[6b] Matriz de información mutua (discretizada)")
		print(info_mutua.round(4).to_string())

		fig, ax = plt.subplots(figsize=(1.2 * len(numeric_cols_corr) + 4, 1.0 * len(numeric_cols_corr) + 3))
		sns.heatmap(info_mutua, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
		ax.set_title(f"Heatmap de información mutua - {nombre_dataset}", fontweight="bold")
		plt.tight_layout()
		path_mi = os.path.join(output_dir, f"informacion_mutua_{nombre_dataset}.png")
		plt.savefig(path_mi, dpi=300, bbox_inches="tight")
		plt.close()
		print(f"Figura de información mutua guardada: {path_mi}")
	else:
		print("\n[6b] Matriz de información mutua")
		print("Se necesitan al menos 2 columnas numéricas para calcular información mutua.")
	#7) Densidades de todas las columnas numéricas
	path_densidades = _plot_densidades_numericas(
		df,
		nombre_dataset,
		output_dir,
		numeric_cols,
		columnas_excluir=columnas_excluir_densidad,)
	print("\n[7] Funciones de densidad")
	if path_densidades is not None:
		print(f"Figura de densidades guardada: {path_densidades}")
	else:
		print("No hay columnas numéricas para graficar densidades.")
	print("\nResumen rápido")
	print(f"Columna con más entropía: {col_max_entropia}")
	if col_mayor_rango is not None and col_menor_rango is not None:
		print(f"Mayor rango: {col_mayor_rango} | Menor rango: {col_menor_rango}")
	return {"shape": df.shape,"nulos_por_columna": nulos_df,"filas_duplicadas": dup_filas,"tipos_datos": dtypes_df,"entropia_por_columna": entropias,"rangos_numericos": rangos_df,"columna_mas_entropia": col_max_entropia,"columna_mayor_rango": col_mayor_rango,"columna_menor_rango": col_menor_rango,"correlaciones": correlaciones,"informacion_mutua": info_mutua,"figura_correlacion": path_corr,"figura_informacion_mutua": path_mi,"figura_densidades": path_densidades,
	}

def calcular_num_comentarios_por_autor(df_comentarios: pd.DataFrame) -> pd.DataFrame:
	"""Devuelve un DataFrame con el número de comentarios por autor.
	"""
	if "author" not in df_comentarios.columns:
		raise ValueError("El DataFrame de comentarios debe tener una columna 'author'.")
	#Usamos un agg explícito para asegurarnos del nombre de la columna
	conteo = (
		df_comentarios
		.groupby("author", as_index=False)
		.agg(num_comments=("author", "size")))
	return conteo

def unir_comentarios_y_autores(
	df_conteo_comentarios: pd.DataFrame, df_autores: pd.DataFrame
) -> pd.DataFrame:
	"""Une las tablas de conteo de comentarios y autores por 'author'."""
	columnas_necesarias = {"author", "num_comments"}
	if not columnas_necesarias.issubset(df_conteo_comentarios.columns):
		raise ValueError(
			"df_conteo_comentarios debe contener las columnas 'author' y 'num_comments'.")
	if "author" not in df_autores.columns:
		raise ValueError("df_autores debe contener la columna 'author'.")
	df_merged = df_conteo_comentarios.merge(df_autores, on="author", how="left")
	#Resolver posibles duplicados de la columna num_comments
	cols = df_merged.columns
	if "num_comments_x" in cols and "num_comments_y" in cols:
		#Preferimos el conteo calculado a partir del CSV de comentarios ("_x")
		df_merged["num_comments"] = df_merged["num_comments_x"]
		df_merged = df_merged.drop(columns=["num_comments_x", "num_comments_y"])
	elif "num_comments_x" in cols:
		df_merged = df_merged.rename(columns={"num_comments_x": "num_comments"})
	elif "num_comments_y" in cols:
		df_merged = df_merged.rename(columns={"num_comments_y": "num_comments"})
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

def tests_significacion_genero(df_merged: pd.DataFrame) -> Dict[str, Any]:
	"""Realiza tests de significación estadística entre géneros."""
	df = df_merged.copy()
	df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce")
	#Filtrar solo m y f para comparación
	df_mf = df[df["gender_clean"].isin(["m", "f"])].copy()

	if len(df_mf) == 0:
		return {"error": "No hay datos suficientes para comparación"}
	m_data = df_mf[df_mf["gender_clean"] == "m"]["num_comments"].dropna()
	f_data = df_mf[df_mf["gender_clean"] == "f"]["num_comments"].dropna()
	resultados = {}
	#Test de Mann-Whitney U (no paramétrico, robusto a outliers)
	if len(m_data) > 0 and len(f_data) > 0:
		u_stat, p_value_mw = stats.mannwhitneyu(m_data, f_data, alternative='two-sided')
		resultados["mann_whitney"] = {
			"u_statistic": u_stat,
			"p_value": p_value_mw,
			"significativo": p_value_mw < 0.05}
		# T-test (paramétrico, asume normalidad)
		t_stat, p_value_t = stats.ttest_ind(m_data, f_data)
		resultados["t_test"] = {
			"t_statistic": t_stat,
			"p_value": p_value_t,
			"significativo": p_value_t < 0.05}
		# Effect size (Cohen's d)
		mean_diff = m_data.mean() - f_data.mean()
		pooled_std = np.sqrt(((len(m_data)-1)*m_data.std()**2 + (len(f_data)-1)*f_data.std()**2) / (len(m_data)+len(f_data)-2))
		cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
		resultados["effect_size"] = {
			"cohens_d": cohens_d,
			"interpretacion": "pequeño" if abs(cohens_d) < 0.5 else "medio" if abs(cohens_d) < 0.8 else "grande"}
	print("\nTests de Significación (m vs f)")
	if "mann_whitney" in resultados:
		print(f"Mann-Whitney U test: p={resultados['mann_whitney']['p_value']:.4f} {'*' if resultados['mann_whitney']['significativo'] else '(no sig.)'}")
	if "t_test" in resultados:
		print(f"T-test: p={resultados['t_test']['p_value']:.4f} {'*' if resultados['t_test']['significativo'] else '(no sig.)'}")
	if "effect_size" in resultados:
		print(f"Cohen's d: {resultados['effect_size']['cohens_d']:.4f} ({resultados['effect_size']['interpretacion']})")
	return resultados

def visualizar_distribucion_genero(df_merged: pd.DataFrame, output_dir: str = "figuras") -> None:
	"""Crea visualizaciones de la distribución de comentarios por género."""
	os.makedirs(output_dir, exist_ok=True)
	df = df_merged.copy()
	df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce")
	#Filtrar solo m y f para visualización principal
	df_mf = df[df["gender_clean"].isin(["m", "f"])].copy()
	#Figura 1: Distribuciones comparativas
	fig, axes = plt.subplots(2, 2, figsize=(15, 12))
	#1.1 Boxplot
	ax = axes[0, 0]
	sns.boxplot(data=df_mf, x="gender_clean", y="num_comments", palette="Set2", ax=ax)
	ax.set_yscale('log')
	ax.set_title("Distribución de Comentarios por Género (escala log)", fontsize=12, fontweight='bold')
	ax.set_xlabel("Género", fontsize=11)
	ax.set_ylabel("Número de Comentarios (log)", fontsize=11)
	#1.2 Violin plot
	ax = axes[0, 1]
	sns.violinplot(data=df_mf, x="gender_clean", y="num_comments", palette="Set2", ax=ax)
	ax.set_yscale('log')
	ax.set_title("Violin Plot - Densidad de Comentarios por Género", fontsize=12, fontweight='bold')
	ax.set_xlabel("Género", fontsize=11)
	ax.set_ylabel("Número de Comentarios (log)", fontsize=11)
	#1.3 Histogramas superpuestos
	ax = axes[1, 0]
	for gender, color in zip(['m', 'f'], ['#66c2a5', '#fc8d62']):
		data = df_mf[df_mf["gender_clean"] == gender]["num_comments"]
		ax.hist(data, bins=50, alpha=0.6, label=f"{gender} (n={len(data)})", color=color, edgecolor='black')
	ax.set_xlabel("Número de Comentarios", fontsize=11)
	ax.set_ylabel("Frecuencia", fontsize=11)
	ax.set_title("Histograma de Comentarios por Género", fontsize=12, fontweight='bold')
	ax.legend()
	ax.set_xlim(0, df_mf["num_comments"].quantile(0.95))  # Truncar outliers extremos
	#1.4 Barras con estadísticas
	ax = axes[1, 1]
	stats_data = df_mf.groupby("gender_clean")["num_comments"].agg(['mean', 'median']).reset_index()
	x = np.arange(len(stats_data))
	width = 0.35
	ax.bar(x - width/2, stats_data['mean'], width, label='Media', color='#8da0cb', edgecolor='black')
	ax.bar(x + width/2, stats_data['median'], width, label='Mediana', color='#e78ac3', edgecolor='black')
	ax.set_xlabel("Género", fontsize=11)
	ax.set_ylabel("Número de Comentarios", fontsize=11)
	ax.set_title("Media vs Mediana por Género", fontsize=12, fontweight='bold')
	ax.set_xticks(x)
	ax.set_xticklabels(stats_data['gender_clean'])
	ax.legend()
	ax.grid(axis='y', alpha=0.3)
	plt.tight_layout()
	fig_path = os.path.join(output_dir, "distribucion_genero.png")
	plt.savefig(fig_path, dpi=300, bbox_inches='tight')
	print(f"\nFigura guardada: {fig_path}")
	plt.close()
	#Figura 2: Distribución de autores y comentarios totales
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	#2.1 Número de autores por género
	ax = axes[0]
	autores_count = df.groupby("gender_clean")["author"].nunique()
	colors = ['#66c2a5', '#fc8d62', '#8da0cb']
	autores_count.plot(kind='bar', ax=ax, color=colors[:len(autores_count)], edgecolor='black')
	ax.set_title("Número de Autores por Género", fontsize=12, fontweight='bold')
	ax.set_xlabel("Género", fontsize=11)
	ax.set_ylabel("Número de Autores", fontsize=11)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
	for i, v in enumerate(autores_count):
		ax.text(i, v + autores_count.max()*0.02, str(v), ha='center', va='bottom', fontweight='bold')
	#2.2 Total de comentarios por género
	ax = axes[1]
	comentarios_total = df.groupby("gender_clean")["num_comments"].sum()
	comentarios_total.plot(kind='bar', ax=ax, color=colors[:len(comentarios_total)], edgecolor='black')
	ax.set_title("Total de Comentarios por Género", fontsize=12, fontweight='bold')
	ax.set_xlabel("Género", fontsize=11)
	ax.set_ylabel("Total de Comentarios", fontsize=11)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
	for i, v in enumerate(comentarios_total):
		ax.text(i, v + comentarios_total.max()*0.02, f"{v:,.0f}", ha='center', va='bottom', fontweight='bold')
	plt.tight_layout()
	fig_path = os.path.join(output_dir, "resumen_autores_comentarios.png")
	plt.savefig(fig_path, dpi=300, bbox_inches='tight')
	print(f"Figura guardada: {fig_path}")
	plt.close()
	print(f"\nTodas las figuras guardadas en: {output_dir}/")

def analisis_exploratorio_por_genero(df_merged: pd.DataFrame, con_graficos: bool = True) -> Dict[str, Any]:
	"""Realiza un análisis exploratorio completo centrado en género.
	"""
	if "num_comments" not in df_merged.columns:
		raise ValueError("df_merged debe tener la columna 'num_comments'.")
	if "gender_clean" not in df_merged.columns:
		raise ValueError("df_merged debe tener la columna 'gender_clean'. Usa normalizar_genero primero.")
	df = df_merged.copy()
	#Aseguramos tipo numérico de num_comments
	df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce")
	#Resumen de autores por género
	autores_por_genero = df.groupby("gender_clean")["author"].nunique().rename("num_authors")
	#Estadísticos descriptivos de num_comments por género
	desc_por_genero = (
		df.groupby("gender_clean")["num_comments"]
		.agg(["count", "mean", "median", "std", "min", "max"]))
	#Moda de num_comments por género
	moda_por_genero = (
		df.groupby("gender_clean")["num_comments"]
		.apply(_moda_serie)
		.rename("mode")
	)
	#Sumamos nº total de comentarios por género (suma de num_comments)
	total_comentarios_por_genero = (
		df.groupby("gender_clean")["num_comments"].sum().rename("total_comments")
	)
	#Unimos todo en una única tabla
	resumen_genero = (
		desc_por_genero
		.join(moda_por_genero)
		.join(autores_por_genero)
		.join(total_comentarios_por_genero)
		.reset_index()
	)
	print("Análisis Exploratorio de Datos - Género")
	print("\nResumen por Género (a nivel de autor)")
	print(resumen_genero.to_string(index=False))
	# También podemos sacar estadísticas globales (sin agrupar) de num_comments
	print("\nEstadísticas Globales de num_comments")
	print(df["num_comments"].describe())

	# Tests de significación
	tests_results = tests_significacion_genero(df)
	# Visualizaciones
	if con_graficos:
		visualizar_distribucion_genero(df)
	resultados: Dict[str, Any] = {
		"resumen_por_genero": resumen_genero,
		"autores_por_genero": autores_por_genero,
		"total_comentarios_por_genero": total_comentarios_por_genero,
		"estadisticas_globales_num_comments": df["num_comments"].describe(),
		"tests_significacion": tests_results,
	}
	return resultados

def preparar_dataset_para_sae(
	path_comentarios: str, 
	path_autores: str,
	max_comments: Optional[int] = None,
	solo_genero_conocido: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Prepara dataset unificado para usar en SAE o clasificador.
		
	"""
	print("Preparación de Dataset para SAE/Clasificador")
	# Cargar datos
	print("\n1. Cargando comentarios...")
	df_comentarios = cargar_comentarios(path_comentarios, nrows=max_comments)
	print(f"{len(df_comentarios):,} comentarios cargados")
	
	print("\n2. Cargando autores...")
	df_autores_raw = cargar_autores(path_autores)
	df_autores = normalizar_genero(df_autores_raw)
	print(f"{len(df_autores):,} autores cargados")
	
	#unir comentarios con género
	print("\n3. Uniendo comentarios con género por autor...")
	df_comentarios_con_genero = df_comentarios.merge(
		df_autores[["author", "gender_clean"]], 
		on="author", 
		how="inner"
	)
	print(f"{len(df_comentarios_con_genero):,} comentarios con info de género")
	#filtrar solo género conocido si se solicita
	if solo_genero_conocido:
		antes = len(df_comentarios_con_genero)
		df_comentarios_con_genero = df_comentarios_con_genero[
			df_comentarios_con_genero["gender_clean"].isin(["m", "f"])
		].reset_index(drop=True)
		print(f"\n4. Filtrando solo género conocido (m/f)...")
		print(f"{len(df_comentarios_con_genero):,} comentarios con género m/f")
		print(f"(Eliminados {antes - len(df_comentarios_con_genero):,} con género unknown)")
	
	#mostrar distribución
	print("\n5. Distribución por género:")
	dist = df_comentarios_con_genero["gender_clean"].value_counts()
	for gender, count in dist.items():
		pct = 100 * count / len(df_comentarios_con_genero)
		print(f"{gender}: {count:,} comentarios ({pct:.1f}%)")
	
	print("Dataset preparado y listo para usar")
	
	return df_comentarios_con_genero, df_autores


def preparar_dataset_para_edad(
	path_comentarios: str,
	path_autores: str,
	max_comments: Optional[int] = None,
	solo_edad_conocida: bool = True,
	age_bins: Optional[list] = None,
	age_labels: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Prepara dataset unificado para clasificación por rangos de edad."""
	if age_bins is None:
		age_bins = DEFAULT_AGE_BINS
	if age_labels is None:
		age_labels = DEFAULT_AGE_GROUP_LABELS
	print("Preparación de Dataset para Clasificación de Edad")

	print("\n1. Cargando comentarios...")
	df_comentarios = cargar_comentarios(path_comentarios, nrows=max_comments)
	print(f"{len(df_comentarios):,} comentarios cargados")

	print("\n2. Cargando autores...")
	df_autores_raw = cargar_autores(path_autores)
	df_autores = normalizar_edad(
		df_autores_raw,
		age_bins=age_bins,
		age_labels=age_labels,
	)
	print(f"{len(df_autores):,} autores cargados")

	print("\n3. Uniendo comentarios con edad por autor...")
	df_comentarios_con_edad = df_comentarios.merge(
		df_autores[["author", "age_clean", "age_group"]],
		on="author",
		how="inner",
	)
	print(f"{len(df_comentarios_con_edad):,} comentarios con info de edad")

	if solo_edad_conocida:
		antes = len(df_comentarios_con_edad)
		df_comentarios_con_edad = df_comentarios_con_edad[
			df_comentarios_con_edad["age_group"].isin(age_labels)
		].reset_index(drop=True)
		print("\n4. Filtrando solo edad conocida (rangos válidos)...")
		print(f"{len(df_comentarios_con_edad):,} comentarios con rango de edad válido")
		print(
			f"(Eliminados {antes - len(df_comentarios_con_edad):,} comentarios con edad nula/fuera de rango)"
		)

	print("\n5. Distribución por rango de edad:")
	dist = df_comentarios_con_edad["age_group"].value_counts()
	for age_group, count in dist.items():
		pct = 100 * count / len(df_comentarios_con_edad)
		print(f"{age_group}: {count:,} comentarios ({pct:.1f}%)")

	print("Dataset de edad preparado y listo para usar")

	return df_comentarios_con_edad, df_autores


# Columnas relevantes para el experimento
COLUMNAS_EXPERIMENTO = ["age", "gender", "perceiving", "introverted", "thinking", "intuitive"]
COLUMNAS_MBTI = ["perceiving", "introverted", "thinking", "intuitive"]

def preparar_dataset_para_mbti(
	path_comentarios: str,
	path_autores: str,
	columna_mbti: str,
	max_comments: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Prepara dataset para clasificacion binaria de una dimension MBTI.
	"""

	print(f"Preparación de Dataset para Clasificación de {columna_mbti.upper()}")

	print("\n1. Cargando comentarios...")
	df_comentarios = cargar_comentarios(path_comentarios, nrows=max_comments)
	print(f"{len(df_comentarios):,} comentarios cargados")

	print("\n2. Cargando autores...")
	df_autores = cargar_autores(path_autores)
	print(f"{len(df_autores):,} autores cargados")

	print(f"\n3. Filtrando autores con '{columna_mbti}' conocido...")
	if columna_mbti not in df_autores.columns:
		raise ValueError(f"Columna '{columna_mbti}' no encontrada en autores. "
						 f"Columnas disponibles: {list(df_autores.columns)}")

	df_autores_filtrado = df_autores[df_autores[columna_mbti].notna()].copy()
	df_autores_filtrado[columna_mbti] = df_autores_filtrado[columna_mbti].astype(int)
	n_filtrados = len(df_autores) - len(df_autores_filtrado)
	print(f"{len(df_autores_filtrado):,} autores con {columna_mbti} conocido")
	print(f"(Eliminados {n_filtrados:,} autores con {columna_mbti} nulo)")

	print(f"\n4. Uniendo comentarios con {columna_mbti} por autor...")
	df_merged = df_comentarios.merge(
		df_autores_filtrado[["author", columna_mbti]],
		on="author",
		how="inner",
	)
	print(f"{len(df_merged):,} comentarios con info de {columna_mbti}")

	eliminados = len(df_comentarios) - len(df_merged)
	print(f"(Eliminados {eliminados:,} comentarios sin {columna_mbti} conocido)")

	print(f"\n5. Distribución de {columna_mbti}:")
	dist = df_merged[columna_mbti].value_counts()
	for val, count in dist.items():
		pct = 100 * count / len(df_merged)
		print(f"{int(val)}: {count:,} comentarios ({pct:.1f}%)")

	print(f"Dataset de {columna_mbti} preparado y listo para usar")

	return df_merged, df_autores_filtrado
def analizar_columnas_experimento(
	df_autores: pd.DataFrame,
	output_dir: str = "figuras",
) -> Dict[str, Any]:
	"""Analiza las columnas usadas en los experimentos: edad, genero, MBTI.
	"""
	os.makedirs(output_dir, exist_ok=True)
	print("Análisis de Columnas del Experimento")
	cols_presentes = [c for c in COLUMNAS_EXPERIMENTO if c in df_autores.columns]
	cols_faltantes = [c for c in COLUMNAS_EXPERIMENTO if c not in df_autores.columns]
	if cols_faltantes:
		print(f"AVISO: columnas no encontradas: {cols_faltantes}")
	n_total = len(df_autores)
	resumen = {}

	#Tabla de valores posibles y porcentajes para cada columna
	for col in cols_presentes:
		serie = df_autores[col]
		n_nulos = int(serie.isna().sum())
		pct_nulos = 100 * n_nulos / n_total if n_total > 0 else 0

		print(f"\n{'─' * 60}")
		print(f"Columna: {col}")
		print(f"{'─' * 60}")
		print(f"Total registros: {n_total:,}")
		print(f"Nulos: {n_nulos:,} ({pct_nulos:.1f}%)")
		print(f"No nulos: {n_total - n_nulos:,} ({100 - pct_nulos:.1f}%)")
		print(f"Valores únicos: {serie.nunique(dropna=True)}")

		# Tabla de frecuencias
		vc = serie.value_counts(dropna=False).reset_index()
		vc.columns = ["valor", "conteo"]
		vc["porcentaje"] = (100 * vc["conteo"] / n_total).round(2)
		vc["pct_no_nulos"] = np.where(
			vc["valor"] is not pd.NA,
			(100 * vc["conteo"] / max(n_total - n_nulos, 1)).round(2),
			0.0,
		)

		# Para columnas con muchos valores, mostrar solo top 20
		if len(vc) > 25:
			print(f"\n  Top 20 valores (de {len(vc)} únicos):")
			print(vc.head(20).to_string(index=False))
			# Agrupar el resto
			resto = vc.iloc[20:]
			print(f"... y {len(resto)} valores más ({resto['conteo'].sum():,} registros, "
				  f"{resto['porcentaje'].sum():.1f}%)")
		else:
			print(f"\n  Distribución completa:")
			print(vc.to_string(index=False))

		resumen[col] = vc

	# Outliers para 'age' (numérica continua)
	if "age" in df_autores.columns:
		print(f"\n{'─' * 60}")
		print("Análisis de Outliers: age")
		print(f"{'─' * 60}")
		age = pd.to_numeric(df_autores["age"], errors="coerce").dropna()
		if len(age) > 0:
			q1 = age.quantile(0.25)
			q3 = age.quantile(0.75)
			iqr = q3 - q1
			lower = q1 - 1.5 * iqr
			upper = q3 + 1.5 * iqr
			outliers = age[(age < lower) | (age > upper)]
			print(f"Q1={q1:.1f}  Q3={q3:.1f}  IQR={iqr:.1f}")
			print(f"Límite inferior: {lower:.1f}")
			print(f"Límite superior: {upper:.1f}")
			print(f"Outliers: {len(outliers):,} ({100*len(outliers)/len(age):.1f}%)")
			print(f"Rango de outliers: [{outliers.min():.1f}, {outliers.max():.1f}]")
			print(f"Media: {age.mean():.1f}  Mediana: {age.median():.1f}  Std: {age.std():.1f}")
	#GRÁFICO COMBINAD
	#Layout: 2 filas
	#Fila 1 (3 cols): boxplot edad, histograma edad con IQR, gender
	#Fila 2 (4 cols): 4 dimensiones MBTI
	mbti_presentes = [c for c in COLUMNAS_MBTI if c in df_autores.columns]
	fig = plt.figure(figsize=(16, 7))
	gs = fig.add_gridspec(2, 1, hspace=0.45)
	gs_row1 = gs[0].subgridspec(1, 3, wspace=0.3)
	n_mbti_cols = max(len(mbti_presentes), 1)
	gs_row2 = gs[1].subgridspec(1, n_mbti_cols, wspace=0.3)
	#Fila 1: age boxplot + age histogram + gender
	if "age" in df_autores.columns:
		age = pd.to_numeric(df_autores["age"], errors="coerce").dropna()
		if len(age) > 0:
			#Boxplot
			ax = fig.add_subplot(gs_row1[0])
			ax.boxplot(age, vert=True, patch_artist=True,
					   boxprops=dict(facecolor="#8da0cb", alpha=0.7),
					   medianprops=dict(color="red", linewidth=2))
			ax.set_title("Boxplot de Edad (outliers IQR)", fontsize=12, fontweight="bold")
			ax.set_ylabel("Edad")
			ax.set_xticklabels(["age"])
			#Histograma con líneas de IQR
			ax = fig.add_subplot(gs_row1[1])
			ax.hist(age, bins=40, color="#66c2a5", edgecolor="black", alpha=0.7)
			q1 = age.quantile(0.25)
			q3 = age.quantile(0.75)
			iqr = q3 - q1
			ax.axvline(q1 - 1.5 * iqr, color="red", linestyle="--", label=f"Lower={q1 - 1.5*iqr:.0f}")
			ax.axvline(q3 + 1.5 * iqr, color="red", linestyle="--", label=f"Upper={q3 + 1.5*iqr:.0f}")
			ax.axvline(age.median(), color="blue", linestyle="-", label=f"Mediana={age.median():.0f}")
			ax.set_title("Distribución de Edad con límites de outliers", fontsize=12, fontweight="bold")
			ax.set_xlabel("Edad")
			ax.set_ylabel("Frecuencia")
			ax.legend()
	if "gender" in df_autores.columns:
		ax = fig.add_subplot(gs_row1[2])
		vc = df_autores["gender"].value_counts(dropna=False).head(10)
		labels = [str(v) if pd.notna(v) else "NaN" for v in vc.index]
		colors_bar = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
		bars = ax.bar(labels, vc.values, color=colors_bar[:len(labels)], edgecolor="black")
		ax.set_title("Distribución género", fontsize=12, fontweight="bold")
		ax.set_xlabel("Gender")
		ax.set_ylabel("Nº de autores")
		for bar, val in zip(bars, vc.values):
			pct = 100 * val / n_total
			ax.text(bar.get_x() + bar.get_width() / 2, val + n_total * 0.01,
					f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

	# Fila 2: Dimensiones MBTI (4 columnas)
	if mbti_presentes:
		palette = {"0": "#fc8d62", "1": "#66c2a5", "NaN": "#cccccc"}
		for idx, col in enumerate(mbti_presentes):
			ax = fig.add_subplot(gs_row2[idx])

			serie = df_autores[col]
			vc = serie.value_counts(dropna=False)
			bar_labels = []
			vals = []
			bar_colors = []
			for v in [1.0, 0.0]:
				if v in vc.index:
					bar_labels.append(str(int(v)))
					vals.append(vc[v])
					bar_colors.append(palette[str(int(v))])
			n_nan = int(serie.isna().sum())
			if n_nan > 0:
				bar_labels.append("NaN")
				vals.append(n_nan)
				bar_colors.append(palette["NaN"])
			bars = ax.bar(bar_labels, vals, color=bar_colors, edgecolor="black")
			ax.set_title(f"{col}", fontsize=12, fontweight="bold")
			ax.set_ylabel("Nº de autores")
			for bar, val in zip(bars, vals):
				pct = 100 * val / n_total
				ax.text(bar.get_x() + bar.get_width() / 2, val + n_total * 0.008,
						f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
	fig.suptitle("Análisis de columnas del experimento", fontsize=15, fontweight="bold", y=1.01)
	path_fig = os.path.join(output_dir, "columnas_experimento.png")
	plt.savefig(path_fig, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"\nFigura combinada guardada: {path_fig}")
	print(f"\nAnálisis de columnas del experimento completado")
	return resumen

if __name__ == "__main__":
	print("\n" + "#"*60)
	print("Preprocesamiento Y Análisis Exploratorio de Datos")
	#rutas a los datos
	path_comentarios = "data/all_comments_since_2015.csv"
	path_autores = "data/author_profiles.csv"
	#1. Cargar datos (puedes quitar nrows para usar todo el dataset)
	print("\n[PASO 1/3] Cargando datos...")
	df_comentarios = cargar_comentarios(path_comentarios, nrows=None)
	df_autores_raw = cargar_autores(path_autores)
	#1b. Análisis general inicial de calidad/estructura por dataset
	print("\n[PASO 1b] Análisis general inicial de cada dataset...")
	analizar_dataset_general(df_comentarios, nombre_dataset="comentarios", output_dir="figuras")
	columnas_excluir_densidad_autores = ["is_female_proba","is_female_pred","is_female_predicted_test","is_native_english_country","eneagram_wing","eneagram_type","en_comments_percentage","en_comments","is_score","is_description",]
	analizar_dataset_general(
		df_autores_raw,
		nombre_dataset="autores",
		output_dir="figuras",
		columnas_excluir_densidad=columnas_excluir_densidad_autores,
		columnas_excluir_correlacion=["is_description","is_score","is_female_proba","is_female_pred","is_female","predicted_test","is_female_predicted_test","is_native_english_country","enneagram_wing","enneagram_type","eneagram_wing","eneagram_type","en_comments_percentage",
		],
	)
	#2. Normalizar género en la tabla de autores
	print("\n[PASO 2/3] Normalizando género...")
	df_autores = normalizar_genero(df_autores_raw)
	#3. Calcular nº de comentarios por autor
	print("\n[PASO 3/3] Calculando estadísticas por autor...")
	df_conteo = calcular_num_comentarios_por_autor(df_comentarios)
	#4. Unir conteo con info de autores (incluyendo gender_clean)
	df_merged = unir_comentarios_y_autores(df_conteo, df_autores)
	#5. Análisis exploratorio centrado en género CON GRÁFICOS
	resultados = analisis_exploratorio_por_genero(df_merged, con_graficos=True)
	#. Análisis de columnas del experimento (edad, género, country, MBTI)
	print("\n[PASO 6] Análisis de columnas del experimento...")
	analizar_columnas_experimento(df_autores_raw, output_dir="figuras")
	print("\n" + "#"*60)
	print("Análisis Completo")
	print("\nAnálisis exploratorio completado con éxito")
	print("Gráficos guardados en: figuras/")
	print("\nPara usar este dataset en SAE/clasificador:")
	print("from preprocesamiento import preparar_dataset_para_sae")
	print("df_comentarios, df_autores = preparar_dataset_para_sae(...)")

