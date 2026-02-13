# Mejores Prácticas de Análisis Exploratorio de Datos (EDA)

**Proyecto**: Gestion_Datos — Maestría en Ciencia de Datos
**Stack**: Python 3.9+, pandas, numpy, matplotlib, seaborn, scikit-learn
**Última actualización**: 2026-02-11

---

## Índice

1. [Filosofía del EDA](#1-filosofia-del-eda)
2. [Checklist Completo del EDA](#2-checklist-completo-del-eda)
3. [Carga y Validación Inicial](#3-carga-y-validacion-inicial)
4. [Análisis de Calidad de Datos](#4-analisis-de-calidad-de-datos)
5. [Estadísticas Descriptivas](#5-estadisticas-descriptivas)
6. [Visualizaciones Recomendadas](#6-visualizaciones-recomendadas)
7. [Análisis de Correlaciones](#7-analisis-de-correlaciones)
8. [Detección de Outliers](#8-deteccion-de-outliers)
9. [Análisis de Variables Categóricas](#9-analisis-de-variables-categoricas)
10. [Análisis Temporal](#10-analisis-temporal)
11. [Errores Comunes y Cómo Evitarlos](#11-errores-comunes-y-como-evitarlos)
12. [Patrones de Código Reutilizables](#12-patrones-de-codigo-reutilizables)
13. [Referencias a Módulos del Proyecto](#13-referencias-a-modulos-del-proyecto)

---

## 1. Filosofía del EDA

El EDA no es un paso mecánico: es una **conversación con los datos**. El objetivo es:

- Comprender la estructura y calidad de los datos antes de modelar
- Generar hipótesis y preguntas de negocio
- Identificar problemas (nulos, outliers, inconsistencias) antes de que afecten los modelos
- Comunicar hallazgos a stakeholders técnicos y no técnicos

**Principio fundamental**: Los datos siempre tienen historia. El EDA es el proceso de descubrirla.

### Cuándo termina el EDA

El EDA nunca termina completamente. Es iterativo: a medida que se modelan los datos y se obtienen nuevos hallazgos, vuelves a explorar. Sin embargo, un EDA inicial está completo cuando:

1. Conoces la forma, tamaño y tipos de datos
2. Documentaste los patrones de nulos
3. Identificaste outliers y decidiste cómo tratarlos
4. Tienes hipótesis sobre las relaciones entre variables
5. El dataset está en condiciones de entrar al pipeline de features

---

## 2. Checklist Completo del EDA

### Fase 0: Configuración

- [ ] Entorno virtual activado con dependencias instaladas
- [ ] Notebook numerado con prefijo (`01_eda_...ipynb`)
- [ ] Imports organizados: stdlib → third-party → locales
- [ ] Rutas usando `pathlib.Path`, no strings hardcodeados
- [ ] `src/` agregado al `sys.path` para importar módulos del proyecto
- [ ] Estilo visual configurado con `set_style()` de `src/visualization/plots.py`
- [ ] Directorio de figuras creado: `reports/figures/`

### Fase 1: Carga de Datos

- [ ] Archivo cargado correctamente (sin errores silenciosos de encoding)
- [ ] Shape verificada (filas x columnas)
- [ ] Uso de memoria estimado (`df.memory_usage(deep=True).sum()`)
- [ ] Muestra de las primeras y últimas N filas revisada
- [ ] Tipos de datos (`dtypes`) inspeccionados

### Fase 2: Calidad de Datos

- [ ] Conteo de valores nulos por columna (`isnull().sum()`)
- [ ] Porcentaje de nulos por columna calculado
- [ ] Patrón de nulos visualizado (heatmap o missingno matrix)
- [ ] Duplicados contados (por fila completa y por clave natural)
- [ ] Valores vacíos en strings detectados (`df == ''`)
- [ ] Tipos de datos incorrectos identificados (ej: número almacenado como string)
- [ ] Inconsistencias de formato detectadas (fechas en múltiples formatos, etc.)

### Fase 3: Estadísticas Descriptivas

- [ ] `describe()` para variables numéricas analizado
- [ ] `describe()` para variables categóricas (`include='object'`) analizado
- [ ] Distribución de frecuencias de variables categóricas revisada
- [ ] Rango de valores razonables verificado para cada variable clave
- [ ] Variables con varianza cero o muy baja identificadas

### Fase 4: Identificación de Variables

- [ ] Variables clasificadas: numéricas continuas, discretas, booleanas, categóricas, texto, listas, fechas
- [ ] Variable objetivo (target) identificada si aplica
- [ ] Identificadores únicos (IDs) separados de features
- [ ] Variables con alta cardinalidad identificadas

### Fase 5: Visualizaciones

- [ ] Histograma o distribución para cada variable numérica importante
- [ ] Barplot o countplot para variables categóricas
- [ ] Heatmap de correlaciones para variables numéricas
- [ ] Scatter plots para relaciones clave
- [ ] Box/violin plots para comparaciones entre grupos
- [ ] Serie temporal si hay variable de tiempo
- [ ] Todas las figuras exportadas a `reports/figures/`

### Fase 6: Análisis de Outliers

- [ ] Outliers identificados con método IQR o Z-score
- [ ] Boxplots generados para variables con outliers significativos
- [ ] Decisión documentada: eliminar, winsorizar, o mantener outliers

### Fase 7: Análisis Bivariado y Multivariado

- [ ] Correlaciones entre features y target calculadas
- [ ] Relaciones entre features importantes graficadas
- [ ] Interacciones de alto valor identificadas

### Fase 8: Documentación y Conclusiones

- [ ] Hipótesis documentadas en celdas markdown
- [ ] Decisiones de limpieza justificadas
- [ ] Resumen ejecutivo al final del notebook
- [ ] Dataset procesado guardado en `data/processed/`

---

## 3. Carga y Validación Inicial

### Patrón recomendado para carga de datos

```python
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Configurar path del proyecto
PROJECT_ROOT = Path.cwd().parent.parent  # ajustar según profundidad del notebook
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_data
from src.visualization.plots import set_style

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Estilo visual
set_style()

# Rutas con pathlib
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Carga
df = load_data(str(DATA_DIR / 'mi_dataset.csv'))
logger.info(f'Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas')
```

### Inspección inicial estructurada

```python
def quick_inspect(df: pd.DataFrame) -> None:
    """Imprime un resumen rapido del DataFrame.

    Args:
        df: DataFrame a inspeccionar.
    """
    print(f'Shape: {df.shape[0]:,} filas x {df.shape[1]} columnas')
    print(f'Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
    print(f'Duplicados: {df.duplicated().sum():,}')
    print(f'Nulos totales: {df.isnull().sum().sum():,} ({df.isnull().mean().mean()*100:.1f}%)')
    print()
    print('Tipos de datos:')
    print(df.dtypes.value_counts())
    print()
    print('Primeras 3 filas:')
    display(df.head(3))
```

---

## 4. Análisis de Calidad de Datos

### Reporte completo de nulos

```python
def null_report(df: pd.DataFrame) -> pd.DataFrame:
    """Genera un reporte detallado de valores nulos por columna.

    Args:
        df: DataFrame a analizar.

    Returns:
        DataFrame con columnas: nulos, pct_nulos, dtype.
    """
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)

    report = pd.DataFrame({
        'nulos': null_counts,
        'pct_nulos': null_pct,
        'dtype': df.dtypes
    }).sort_values('pct_nulos', ascending=False)

    return report[report['nulos'] > 0]

# Uso
reporte = null_report(df)
print(reporte)
```

### Visualización de patrón de nulos

```python
import missingno as msno

# Opción 1: Missingno matrix (requiere pip install missingno)
fig, ax = plt.subplots(figsize=(14, 6))
msno.matrix(df.sample(min(500, len(df)), random_state=42), ax=ax)
plt.title('Patrón de Valores Nulos')
fig.savefig(FIGURES_DIR / 'heatmap_nulos.png', dpi=150, bbox_inches='tight')

# Opción 2: Heatmap con seaborn (sin dependencias extra)
fig, ax = plt.subplots(figsize=(14, 6))
cols_con_nulos = df.columns[df.isnull().any()].tolist()
sns.heatmap(df[cols_con_nulos].isnull().astype(int).head(100),
            cmap='Blues', cbar=False, ax=ax)
plt.title('Mapa de Nulos (1=nulo, 0=presente)')
fig.savefig(FIGURES_DIR / 'heatmap_nulos.png', dpi=150, bbox_inches='tight')
```

### Estrategias de imputación según tipo de columna

| Tipo de Variable | Estrategia Recomendada | Cuándo Usar |
|---|---|---|
| Numérica con < 5% nulos | `fillna(median)` | Distribución sesgada |
| Numérica con < 5% nulos | `fillna(mean)` | Distribución normal |
| Numérica con > 20% nulos | Indicador de nulo + imputación | Cuando el nulo es informativo |
| Categórica | `fillna('Unknown')` o moda | Si hay categoría dominante |
| Series temporal | `fillna(method='ffill')` | Datos secuenciales |
| Target variable | **Nunca imputar** | Eliminar filas con target nulo |

---

## 5. Estadísticas Descriptivas

### Análisis completo de distribuciones

```python
def describe_extended(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas descriptivas extendidas con skewness y kurtosis.

    Args:
        df: DataFrame con variables numéricas.

    Returns:
        DataFrame con estadísticas por variable.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    stats = numeric_df.describe().T
    stats['skewness'] = numeric_df.skew().round(3)
    stats['kurtosis'] = numeric_df.kurtosis().round(3)
    stats['nulls_pct'] = (numeric_df.isnull().mean() * 100).round(2)
    stats['zeros_pct'] = ((numeric_df == 0).mean() * 100).round(2)

    return stats.sort_values('skewness', key=abs, ascending=False)

# Uso
stats = describe_extended(df)
display(stats)
```

### Interpretación de skewness

- `|skewness| < 0.5`: distribución aproximadamente simétrica
- `0.5 < |skewness| < 1.0`: asimetría moderada, considerar transformación log
- `|skewness| > 1.0`: asimetría severa, **aplicar transformación log o BoxCox**

---

## 6. Visualizaciones Recomendadas

### Uso de los módulos del proyecto

```python
from src.visualization.plots import set_style, plot_distribution, plot_correlation

# Configurar estilo global
set_style()  # seaborn-v0_8-darkgrid + paleta husl

# Distribución de una variable
plot_distribution(df['price'], title='Distribución de Precios', kde=True, bins=50)

# Correlación entre numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns
plot_correlation(df[numeric_cols], title='Matriz de Correlaciones')
```

### Exportar figuras correctamente

```python
# SIEMPRE exportar antes de plt.show()
fig, ax = plt.subplots(figsize=(12, 6))
# ... tu visualización aquí ...
plt.tight_layout()

# Guardar en reports/figures/ con nombre descriptivo
output_path = FIGURES_DIR / 'nombre_descriptivo_viz.png'
fig.savefig(output_path, dpi=150, bbox_inches='tight')
plt.show()
logger.info(f'Figura guardada: {output_path}')
```

### Paleta de colores del proyecto

```python
# Paleta husl (por defecto con set_style())
palette = sns.color_palette('husl', n_colors=10)

# Para variables categóricas
sns.barplot(data=df, x='categoria', y='valor', palette='husl')

# Para mapas de calor (correlaciones)
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)

# Para distribuciones
sns.histplot(df['variable'], color=palette[0], kde=True)
```

### Visualizaciones por tipo de análisis

#### Univariado (una variable)

```python
# Numérica continua
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['variable'], bins=50, edgecolor='white')
axes[0].set_title('Histograma')
sns.boxplot(y=df['variable'], ax=axes[1])
axes[1].set_title('Boxplot')
fig.savefig(FIGURES_DIR / 'univariado_variable.png', dpi=150, bbox_inches='tight')

# Categórica
top_cats = df['categoria'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(12, 6))
top_cats.plot(kind='barh', ax=ax)
ax.set_title('Top 15 Categorías')
fig.savefig(FIGURES_DIR / 'categorica_top15.png', dpi=150, bbox_inches='tight')
```

#### Bivariado (dos variables)

```python
# Numérica vs Numérica
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(df['x'], df['y'], alpha=0.3, s=10)
ax.set_xlabel('Variable X')
ax.set_ylabel('Variable Y')
fig.savefig(FIGURES_DIR / 'scatter_x_vs_y.png', dpi=150, bbox_inches='tight')

# Categórica vs Numérica
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x='categoria', y='numerica', palette='husl', ax=ax, showfliers=False)
plt.xticks(rotation=45)
fig.savefig(FIGURES_DIR / 'boxplot_cat_vs_num.png', dpi=150, bbox_inches='tight')

# Categórica vs Categórica
cross_tab = pd.crosstab(df['cat1'], df['cat2'], normalize='index')
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(cross_tab, annot=True, fmt='.1%', cmap='Blues', ax=ax)
fig.savefig(FIGURES_DIR / 'crosstab_cat1_cat2.png', dpi=150, bbox_inches='tight')
```

---

## 7. Análisis de Correlaciones

### Correlación de Pearson vs Spearman

```python
# Pearson: asume linealidad y normalidad
corr_pearson = df[numeric_cols].corr(method='pearson')

# Spearman: no paramétrica, robusta a outliers y relaciones monotónicas
corr_spearman = df[numeric_cols].corr(method='spearman')

# Comparar diferencias
diff = (corr_pearson - corr_spearman).abs()
print('Variables con mayor diferencia Pearson vs Spearman:')
print(diff.unstack().nlargest(10))
```

### Identificar correlaciones significativas

```python
def significant_correlations(
    df: pd.DataFrame,
    threshold: float = 0.3,
    method: str = 'spearman'
) -> pd.DataFrame:
    """Retorna pares de variables con correlación significativa.

    Args:
        df: DataFrame con variables numéricas.
        threshold: Umbral absoluto de correlación para considerar significativa.
        method: Metodo de correlacion ('pearson', 'spearman', 'kendall').

    Returns:
        DataFrame con pares ordenados por correlacion descendente.
    """
    corr = df.select_dtypes(include=[np.number]).corr(method=method)

    pairs = (
        corr.unstack()
        .reset_index()
        .rename(columns={'level_0': 'var1', 'level_1': 'var2', 0: 'corr'})
    )
    pairs = pairs[pairs['var1'] < pairs['var2']]  # evitar duplicados
    pairs = pairs[pairs['corr'].abs() >= threshold]

    return pairs.sort_values('corr', key=abs, ascending=False)

# Uso
sig_corr = significant_correlations(df, threshold=0.3)
print(sig_corr.head(20))
```

---

## 8. Detección de Outliers

### Método IQR (robusto, recomendado)

```python
def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> dict:
    """Detecta outliers usando el método IQR.

    Args:
        series: Serie numérica.
        factor: Factor multiplicador del IQR (1.5 = outliers, 3.0 = extremos).

    Returns:
        Diccionario con límites e índices de outliers.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR

    outlier_mask = (series < lower) | (series > upper)

    return {
        'lower_bound': lower,
        'upper_bound': upper,
        'n_outliers': outlier_mask.sum(),
        'pct_outliers': outlier_mask.mean() * 100,
        'outlier_indices': series[outlier_mask].index.tolist()
    }
```

### Estrategias de tratamiento de outliers

```python
# Opción 1: Winsorizing (recorte a percentiles)
from scipy.stats import mstats
df['variable_winsorized'] = mstats.winsorize(df['variable'], limits=[0.01, 0.01])

# Opción 2: Transformación logarítmica (reduce impacto)
df['variable_log'] = np.log1p(df['variable'])  # log1p evita log(0)

# Opción 3: Capping manual a límites de negocio
PRICE_MAX = 200.0  # precio máximo razonable para análisis
df['price_capped'] = df['price'].clip(0, PRICE_MAX)

# Opción 4: Eliminar outliers (solo si justificado)
# DOCUMENTAR siempre cuántas filas se eliminan y por qué
mask_clean = (df['variable'] >= lower_bound) & (df['variable'] <= upper_bound)
df_clean = df[mask_clean].copy()
logger.info(f'Filas eliminadas por outliers: {(~mask_clean).sum():,}')
```

---

## 9. Análisis de Variables Categóricas

### Evaluación de cardinalidad

```python
def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Genera un resumen de variables categóricas con cardinalidad.

    Args:
        df: DataFrame a analizar.

    Returns:
        DataFrame con estadísticas por variable categórica.
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    summary = []
    for col in cat_cols:
        n_unique = df[col].nunique()
        top_value = df[col].value_counts().index[0] if n_unique > 0 else None
        top_pct = df[col].value_counts(normalize=True).iloc[0] * 100 if n_unique > 0 else 0

        summary.append({
            'columna': col,
            'n_unique': n_unique,
            'top_value': top_value,
            'top_pct': round(top_pct, 2),
            'pct_nulos': round(df[col].isnull().mean() * 100, 2)
        })

    return pd.DataFrame(summary).sort_values('n_unique', ascending=False)
```

### Encoding según cardinalidad

| Cardinalidad | Estrategia | Ejemplo |
|---|---|---|
| 2 valores (binaria) | Label Encoding (0/1) | `True/False`, `Si/No` |
| 2-10 valores ordenados | Label Encoding ordinal | `Low/Mid/High` |
| 3-15 valores sin orden | One-Hot Encoding | Géneros, plataformas |
| > 15 valores | Frequency Encoding o Target Encoding | Developers, publishers |
| > 100 valores (ID-like) | Descartar o usar como embedding | Nombres de juegos |

---

## 10. Análisis Temporal

### Buenas prácticas con fechas

```python
# Parseo robusto de fechas
df['fecha'] = pd.to_datetime(df['fecha'], format='mixed', errors='coerce')

# Verificar cobertura temporal
n_validas = df['fecha'].notna().sum()
print(f'Fechas válidas: {n_validas:,} ({n_validas/len(df)*100:.1f}%)')
print(f'Rango: {df["fecha"].min()} a {df["fecha"].max()}')

# Extraer componentes
df['year'] = df['fecha'].dt.year
df['month'] = df['fecha'].dt.month
df['quarter'] = df['fecha'].dt.quarter
df['day_of_week'] = df['fecha'].dt.dayofweek  # 0=Lunes, 6=Domingo
df['is_weekend'] = df['day_of_week'].isin([5, 6])

# Antigüedad calculada correctamente
REFERENCE_DATE = pd.Timestamp.now()
df['age_days'] = (REFERENCE_DATE - df['fecha']).dt.days
df['age_years'] = df['age_days'] / 365.25
```

### Visualización de series temporales

```python
# Agrupación por período
monthly = df.set_index('fecha').resample('ME').size()

fig, ax = plt.subplots(figsize=(14, 5))
monthly.plot(ax=ax, linewidth=2, color='#2196F3')
ax.fill_between(monthly.index, monthly.values, alpha=0.2, color='#2196F3')
ax.set_title('Evolución Temporal')
ax.set_xlabel('Fecha')
ax.set_ylabel('Conteo')
fig.savefig(FIGURES_DIR / 'serie_temporal.png', dpi=150, bbox_inches='tight')
```

---

## 11. Errores Comunes y Cómo Evitarlos

### Error 1: No verificar tipos de datos al cargar

**Problema**: `pd.read_csv()` puede inferir tipos incorrectamente (ej: IDs numéricos leídos como int).

```python
# INCORRECTO
df = pd.read_csv('data.csv')
# Si 'app_id' tiene valores como '0001234', se perderán los ceros

# CORRECTO
df = pd.read_csv('data.csv', dtype={'app_id': str, 'codigo_postal': str})
```

### Error 2: Imputar antes de analizar

**Problema**: Imputar valores nulos antes de entender su patrón puede introducir sesgos.

```python
# INCORRECTO (imputación ciega)
df['score'].fillna(df['score'].mean(), inplace=True)

# CORRECTO (analizar primero)
# 1. Entender por qué hay nulos (¿MCAR, MAR, MNAR?)
# 2. Visualizar el patrón
# 3. Decidir estrategia con evidencia
print(f"Nulos en score: {df['score'].isnull().sum()} ({df['score'].isnull().mean():.1%})")
# Analizar si los nulos están correlacionados con otras variables
print(df.groupby(df['score'].isnull())['otras_variables'].mean())
```

### Error 3: Olvidar el efecto de los outliers en estadísticas

**Problema**: La media es muy sensible a outliers. Reportarla sin el contexto de outliers es engañoso.

```python
# INCORRECTO: reportar solo la media
print(f"Precio promedio: ${df['price'].mean():.2f}")

# CORRECTO: reportar mediana y percentiles junto con media
print(f"Precio - Media: ${df['price'].mean():.2f}")
print(f"Precio - Mediana: ${df['price'].median():.2f}")
print(f"Precio - P25/P75: ${df['price'].quantile(0.25):.2f} / ${df['price'].quantile(0.75):.2f}")
print(f"Precio - Max: ${df['price'].max():.2f}")
```

### Error 4: Correlación sin causalidad

**Problema**: Una correlación alta no implica relación causal. Siempre contextualizar.

```python
# Calcular correlación
corr = df['n_reviews'].corr(df['price'])
print(f'Correlación reviews-precio: {corr:.3f}')

# SIEMPRE acompañar con visualización e interpretación
# No concluir "precio alto CAUSA más reviews"
# Puede ser: juegos más conocidos (causa común) tienen tanto más reviews como precios estándar
```

### Error 5: Hardcodear rutas

**Problema**: El código deja de funcionar en otro sistema o directorio de trabajo.

```python
# INCORRECTO
df = pd.read_csv('C:/Users/Christian/datos.csv')
df.to_csv('../processed/clean.csv')

# CORRECTO
PROJECT_ROOT = Path.cwd().parent.parent
df = load_data(str(PROJECT_ROOT / 'data' / 'raw' / 'datos.csv'))
df.to_csv(PROJECT_ROOT / 'data' / 'processed' / 'clean.csv', index=False)
```

### Error 6: No guardar figuras antes de plt.show()

**Problema**: En Jupyter, `plt.show()` limpia el buffer. Si llamas `savefig()` después, la figura está vacía.

```python
# INCORRECTO
plt.show()
fig.savefig('figura.png')  # Guarda figura vacía

# CORRECTO
fig.savefig(FIGURES_DIR / 'figura.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Error 7: Analizar el dataset completo sin muestreo para visualizaciones

**Problema**: Scatter plots con 100K+ puntos son lentos, ilegibles y sin valor informativo.

```python
# INCORRECTO para datasets grandes
plt.scatter(df['x'], df['y'])  # 100K puntos = figura inútil

# CORRECTO
sample_size = min(2000, len(df))
sample = df.sample(sample_size, random_state=42)
plt.scatter(sample['x'], sample['y'], alpha=0.3, s=10)
plt.title(f'Muestra de {sample_size:,} / {len(df):,} registros')
```

### Error 8: Modificar el dataset raw

**Problema**: Perder los datos originales hace imposible reproducir el análisis desde el principio.

```python
# INCORRECTO
df = load_data(str(DATA_DIR / 'games.csv'))
df.dropna(inplace=True)  # Modifica el df original
df.to_csv(DATA_DIR / 'games.csv')  # ¡Sobreescribe raw!

# CORRECTO
df_raw = load_data(str(DATA_DIR / 'games.csv'))  # raw nunca se toca
df = df_raw.copy()  # trabajar sobre copia
df = df.dropna(subset=['name'])  # evitar inplace=True
df.to_csv(PROCESSED_DIR / 'games_clean.csv', index=False)  # guardar en processed/
```

---

## 12. Patrones de Código Reutilizables

### Guardado automático de figuras con convención de nombres

```python
from datetime import date

def save_figure(fig: plt.Figure, name: str, figures_dir: Path) -> Path:
    """Guarda una figura con nombre estandarizado.

    Convención: {nombre_descriptivo}_{fecha}.png

    Args:
        fig: Figura de matplotlib a guardar.
        name: Nombre descriptivo (sin extensión).
        figures_dir: Directorio destino.

    Returns:
        Path donde se guardó la figura.
    """
    today = date.today().isoformat()
    filename = f'{name}_{today}.png'
    output_path = figures_dir / filename

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f'Figura guardada: {output_path}')

    return output_path
```

### Perfilado completo de un dataset

```python
def profile_dataset(df: pd.DataFrame) -> None:
    """Ejecuta un perfil completo del dataset.

    Args:
        df: DataFrame a perfilar.
    """
    print('=' * 60)
    print('PERFIL DEL DATASET')
    print('=' * 60)
    print(f'Forma: {df.shape[0]:,} filas x {df.shape[1]} columnas')
    print(f'Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB')
    print(f'Duplicados: {df.duplicated().sum():,}')
    print()

    # Por tipo
    type_counts = df.dtypes.value_counts()
    print('Tipos de datos:')
    for dtype, count in type_counts.items():
        print(f'  {dtype}: {count} columnas')
    print()

    # Nulos
    null_pcts = df.isnull().mean() * 100
    cols_with_nulls = null_pcts[null_pcts > 0].sort_values(ascending=False)
    if len(cols_with_nulls) > 0:
        print(f'Columnas con nulos ({len(cols_with_nulls)}):')
        for col, pct in cols_with_nulls.head(10).items():
            print(f'  {col}: {pct:.1f}%')
    else:
        print('Sin valores nulos.')

    print('=' * 60)
```

---

## 13. Referencias a Módulos del Proyecto

### `src/data/loader.py`

Función principal: `load_data(filepath: str) -> pd.DataFrame`

Soporta: CSV, XLSX, JSON (formato plano). Para JSON anidado como `games.json`, usar función personalizada con `json.load()`.

```python
from src.data.loader import load_data

# CSV o XLSX
df = load_data(str(PROJECT_ROOT / 'data' / 'raw' / 'archivo.csv'))

# Para JSON anidado, usar función propia (ver notebook 01)
```

### `src/visualization/plots.py`

Funciones disponibles:
- `set_style(style='seaborn-v0_8-darkgrid')`: Configura el estilo global de matplotlib/seaborn
- `plot_distribution(data, title, **kwargs)`: Histograma de una Serie con seaborn
- `plot_correlation(data, title)`: Heatmap de correlación de un DataFrame

```python
from src.visualization.plots import set_style, plot_distribution, plot_correlation

# Configurar al inicio del notebook
set_style()

# Distribución rápida
plot_distribution(df['price'], title='Distribución de Precios', kde=True)

# Correlaciones
plot_correlation(df[numeric_cols], title='Correlaciones del Dataset')
```

### Módulos pendientes (`src/features/`, `src/models/`)

Actualmente son stubs. El código de feature engineering desarrollado en los notebooks debe migrarse aquí cuando esté estable. Ver `notebooks/exploratory/02_transformaciones_features_steam.ipynb` como referencia.

---

## Checklist Rápido Pre-Commit

Antes de commitear un notebook de EDA, verificar:

- [ ] Kernel reiniciado y ejecutado completamente sin errores (`Kernel > Restart & Run All`)
- [ ] Output de celdas de exploración grande limpiado (opcional, según política del proyecto)
- [ ] Todas las rutas usan `pathlib.Path` relativo a `PROJECT_ROOT`
- [ ] Figuras exportadas a `reports/figures/`
- [ ] Dataset procesado guardado en `data/processed/` con fecha en el nombre
- [ ] Resumen ejecutivo escrito en la última celda markdown
- [ ] No hay credenciales, tokens o datos sensibles en el notebook

---

*Documento mantenido por el equipo de Data Science de Gestion_Datos.*
