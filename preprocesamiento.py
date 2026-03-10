import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


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

	# Usamos un agg explícito para asegurarnos del nombre de la columna
	conteo = (
		df_comentarios
		.groupby("author", as_index=False)
		.agg(num_comments=("author", "size"))
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

	# Resolver posibles duplicados de la columna num_comments
	cols = df_merged.columns
	if "num_comments_x" in cols and "num_comments_y" in cols:
		# Preferimos el conteo calculado a partir del CSV de comentarios ("_x")
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
	
	# Filtrar solo m y f para comparación
	df_mf = df[df["gender_clean"].isin(["m", "f"])].copy()
	
	if len(df_mf) == 0:
		return {"error": "No hay datos suficientes para comparación"}
	
	m_data = df_mf[df_mf["gender_clean"] == "m"]["num_comments"].dropna()
	f_data = df_mf[df_mf["gender_clean"] == "f"]["num_comments"].dropna()
	
	resultados = {}
	
	# Test de Mann-Whitney U (no paramétrico, robusto a outliers)
	if len(m_data) > 0 and len(f_data) > 0:
		u_stat, p_value_mw = stats.mannwhitneyu(m_data, f_data, alternative='two-sided')
		resultados["mann_whitney"] = {
			"u_statistic": u_stat,
			"p_value": p_value_mw,
			"significativo": p_value_mw < 0.05
		}
		
		# T-test (paramétrico, asume normalidad)
		t_stat, p_value_t = stats.ttest_ind(m_data, f_data)
		resultados["t_test"] = {
			"t_statistic": t_stat,
			"p_value": p_value_t,
			"significativo": p_value_t < 0.05
		}
		
		# Effect size (Cohen's d)
		mean_diff = m_data.mean() - f_data.mean()
		pooled_std = np.sqrt(((len(m_data)-1)*m_data.std()**2 + (len(f_data)-1)*f_data.std()**2) / (len(m_data)+len(f_data)-2))
		cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
		resultados["effect_size"] = {
			"cohens_d": cohens_d,
			"interpretacion": "pequeño" if abs(cohens_d) < 0.5 else "medio" if abs(cohens_d) < 0.8 else "grande"
		}
	
	print("\n===== TESTS DE SIGNIFICACIÓN (m vs f) =====")
	if "mann_whitney" in resultados:
		print(f"Mann-Whitney U test: p={resultados['mann_whitney']['p_value']:.4f} {'*' if resultados['mann_whitney']['significativo'] else '(no sig.)'}")
	if "t_test" in resultados:
		print(f"T-test: p={resultados['t_test']['p_value']:.4f} {'*' if resultados['t_test']['significativo'] else '(no sig.)'}")
	if "effect_size" in resultados:
		print(f"Cohen's d: {resultados['effect_size']['cohens_d']:.4f} ({resultados['effect_size']['interpretacion']})")
	
	return resultados


def visualizar_distribucion_genero(df_merged: pd.DataFrame, output_dir: str = "figuras") -> None:
	"""Crea visualizaciones de la distribución de comentarios por género."""
	
	import os
	os.makedirs(output_dir, exist_ok=True)
	
	df = df_merged.copy()
	df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce")
	
	# Filtrar solo m y f para visualización principal
	df_mf = df[df["gender_clean"].isin(["m", "f"])].copy()
	
	# Figura 1: Distribuciones comparativas
	fig, axes = plt.subplots(2, 2, figsize=(15, 12))
	
	# 1.1 Boxplot
	ax = axes[0, 0]
	sns.boxplot(data=df_mf, x="gender_clean", y="num_comments", palette="Set2", ax=ax)
	ax.set_yscale('log')
	ax.set_title("Distribución de Comentarios por Género (escala log)", fontsize=12, fontweight='bold')
	ax.set_xlabel("Género", fontsize=11)
	ax.set_ylabel("Número de Comentarios (log)", fontsize=11)
	
	# 1.2 Violin plot
	ax = axes[0, 1]
	sns.violinplot(data=df_mf, x="gender_clean", y="num_comments", palette="Set2", ax=ax)
	ax.set_yscale('log')
	ax.set_title("Violin Plot - Densidad de Comentarios por Género", fontsize=12, fontweight='bold')
	ax.set_xlabel("Género", fontsize=11)
	ax.set_ylabel("Número de Comentarios (log)", fontsize=11)
	
	# 1.3 Histogramas superpuestos
	ax = axes[1, 0]
	for gender, color in zip(['m', 'f'], ['#66c2a5', '#fc8d62']):
		data = df_mf[df_mf["gender_clean"] == gender]["num_comments"]
		ax.hist(data, bins=50, alpha=0.6, label=f"{gender} (n={len(data)})", color=color, edgecolor='black')
	ax.set_xlabel("Número de Comentarios", fontsize=11)
	ax.set_ylabel("Frecuencia", fontsize=11)
	ax.set_title("Histograma de Comentarios por Género", fontsize=12, fontweight='bold')
	ax.legend()
	ax.set_xlim(0, df_mf["num_comments"].quantile(0.95))  # Truncar outliers extremos
	
	# 1.4 Barras con estadísticas
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
	print(f"\n✓ Figura guardada: {fig_path}")
	plt.close()
	
	# Figura 2: Distribución de autores y comentarios totales
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	
	# 2.1 Número de autores por género
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
	
	# 2.2 Total de comentarios por género
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
	print(f"✓ Figura guardada: {fig_path}")
	plt.close()
	
	print(f"\n✓ Todas las figuras guardadas en: {output_dir}/")


def analisis_exploratorio_por_genero(df_merged: pd.DataFrame, con_graficos: bool = True) -> Dict[str, Any]:
	"""Realiza un análisis exploratorio completo centrado en género.

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

	print("\n" + "="*60)
	print("ANÁLISIS EXPLORATORIO DE DATOS - GÉNERO")
	print("="*60)
	print("\n===== RESUMEN POR GÉNERO (a nivel de autor) =====")
	print(resumen_genero.to_string(index=False))

	# También podemos sacar estadísticas globales (sin agrupar) de num_comments
	print("\n===== ESTADÍSTICAS GLOBALES DE num_comments =====")
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
	
	Parameters
	----------
	path_comentarios : str
		Ruta al CSV de comentarios
	path_autores : str
		Ruta al CSV de autores
	max_comments : Optional[int]
		Limitar número de comentarios (útil para pruebas)
	solo_genero_conocido : bool
		Si True, solo devuelve comentarios con género 'm' o 'f'
		
	Returns
	-------
	df_comentarios_con_genero : pd.DataFrame
		DataFrame con comentarios + género por autor
	df_autores : pd.DataFrame
		DataFrame de autores con género normalizado
	"""
	
	print("\n" + "="*60)
	print("PREPARACIÓN DE DATASET PARA SAE/CLASIFICADOR")
	print("="*60)
	
	# Cargar datos
	print("\n1. Cargando comentarios...")
	df_comentarios = cargar_comentarios(path_comentarios, nrows=max_comments)
	print(f"   ✓ {len(df_comentarios):,} comentarios cargados")
	
	print("\n2. Cargando autores...")
	df_autores_raw = cargar_autores(path_autores)
	df_autores = normalizar_genero(df_autores_raw)
	print(f"   ✓ {len(df_autores):,} autores cargados")
	
	# Unir comentarios con género
	print("\n3. Uniendo comentarios con género por autor...")
	df_comentarios_con_genero = df_comentarios.merge(
		df_autores[["author", "gender_clean"]], 
		on="author", 
		how="inner"
	)
	print(f"   ✓ {len(df_comentarios_con_genero):,} comentarios con info de género")
	
	# Filtrar solo género conocido si se solicita
	if solo_genero_conocido:
		antes = len(df_comentarios_con_genero)
		df_comentarios_con_genero = df_comentarios_con_genero[
			df_comentarios_con_genero["gender_clean"].isin(["m", "f"])
		].reset_index(drop=True)
		print(f"\n4. Filtrando solo género conocido (m/f)...")
		print(f"   ✓ {len(df_comentarios_con_genero):,} comentarios con género m/f")
		print(f"   (Eliminados {antes - len(df_comentarios_con_genero):,} con género unknown)")
	
	# Mostrar distribución
	print("\n5. Distribución por género:")
	dist = df_comentarios_con_genero["gender_clean"].value_counts()
	for gender, count in dist.items():
		pct = 100 * count / len(df_comentarios_con_genero)
		print(f"   {gender}: {count:,} comentarios ({pct:.1f}%)")
	
	print("\n" + "="*60)
	print("✓ Dataset preparado y listo para usar")
	print("="*60)
	
	return df_comentarios_con_genero, df_autores


if __name__ == "__main__":
	print("\n" + "#"*60)
	print("# PREPROCESAMIENTO Y ANÁLISIS EXPLORATORIO DE DATOS")
	print("#"*60)
	
	# Rutas a los datos
	path_comentarios = "data/all_comments_since_2015.csv"
	path_autores = "data/author_profiles.csv"

	# 1. Cargar datos (puedes quitar nrows para usar todo el dataset)
	print("\n[PASO 1/3] Cargando datos...")
	df_comentarios = cargar_comentarios(path_comentarios, nrows=None)
	df_autores_raw = cargar_autores(path_autores)

	# 2. Normalizar género en la tabla de autores
	print("\n[PASO 2/3] Normalizando género...")
	df_autores = normalizar_genero(df_autores_raw)

	# 3. Calcular nº de comentarios por autor
	print("\n[PASO 3/3] Calculando estadísticas por autor...")
	df_conteo = calcular_num_comentarios_por_autor(df_comentarios)

	# 4. Unir conteo con info de autores (incluyendo gender_clean)
	df_merged = unir_comentarios_y_autores(df_conteo, df_autores)

	# 5. Análisis exploratorio centrado en género CON GRÁFICOS
	resultados = analisis_exploratorio_por_genero(df_merged, con_graficos=True)
	
	print("\n" + "#"*60)
	print("# ANÁLISIS COMPLETO")
	print("#"*60)
	print("\n✓ Análisis exploratorio completado con éxito")
	print("✓ Gráficos guardados en: figuras/")
	print("\nPara usar este dataset en SAE/clasificador:")
	print("  from preprocesamiento import preparar_dataset_para_sae")
	print("  df_comentarios, df_autores = preparar_dataset_para_sae(...)")

