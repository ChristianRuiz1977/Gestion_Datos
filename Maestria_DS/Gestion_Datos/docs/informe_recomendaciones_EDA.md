# Informe de Recomendaciones: EDA en Notebooks del Proyecto Steam Games

**Fecha de evaluacion**: 2026-02-13
**Notebooks analizados**: 4
**Dataset**: Steam Games (~122,610 juegos)
**Referencia**: Mejores practicas de pandas y seaborn (Context7 docs)

---

## Resumen Ejecutivo

Se analizaron los 4 notebooks del proyecto (`01_limpieza_exploracion`, `02_transformaciones_features`, `03_visualizaciones_insights`, `04_reporte_final`) contra las mejores practicas actuales de EDA con pandas, seaborn y scikit-learn. El proyecto demuestra un nivel **alto de madurez** en su pipeline de datos, con areas puntuales de mejora que se detallan a continuacion.

### Calificacion General

| Criterio | Puntuacion | Comentario |
|---|:---:|---|
| Estructura y organizacion | 9/10 | Excelente separacion por fase del pipeline |
| Calidad de codigo | 8/10 | Funciones documentadas, type hints consistentes |
| Manejo de datos faltantes | 7/10 | Analisis de nulos presente, pero falta profundidad en MCAR/MAR/MNAR |
| Visualizaciones | 9/10 | 12 figuras bien diseñadas, exportadas correctamente |
| Feature engineering | 9/10 | Amplio set de transformaciones con validaciones post-transformacion |
| Reproducibilidad | 8/10 | Buena deteccion de PROJECT_ROOT, pero hay variantes entre notebooks |
| Documentacion narrativa | 8/10 | Buenos resumenes ejecutivos, faltan hipotesis intermedias |
| Robustez y validacion | 8/10 | Assertions post-transformacion excelentes, pero sin tests unitarios formales |

---

## 1. Lo que se hizo bien

### 1.1 Estructura del pipeline (Excelente)

Los notebooks siguen una progresion logica clara:

```
01_limpieza  -->  02_features  -->  03_visualizaciones  -->  04_reporte
```

Cada notebook tiene un objetivo especifico, un input y output definidos, y un resumen ejecutivo al final. Esto es una mejor practica fundamental que muchos proyectos no implementan.

### 1.2 Funciones reutilizables con docstrings y type hints

Funciones como `null_report()`, `classify_columns()`, `detect_outliers_iqr()`, `compute_review_features()` y `compute_price_features()` estan bien documentadas con docstrings, type hints y siguen PEP 8. Ejemplo destacado:

```python
def detect_outliers_iqr(series: pd.Series) -> dict:
    """Detecta outliers usando el metodo IQR (rango intercuartilico)."""
```

### 1.3 Validaciones post-transformacion (Destacado)

El notebook 02 incluye assertions despues de cada transformacion para verificar integridad de datos:

```python
assert len(result) == len(df), "La transformacion cambio el numero de filas"
assert result['total_reviews'].ge(0).all(), "total_reviews contiene valores negativos"
```

Esto previene errores silenciosos y es una practica de ingenieria de datos de alto nivel.

### 1.4 Uso de modulos centralizados del proyecto

Se importan y usan modulos de `src/`:
- `src/data/loader.py` para carga de datos
- `src/data/steam_transforms.py` para transformaciones especificas
- `src/visualization/plots.py` para estilo visual

Esto cumple la convencion del proyecto de no escribir logica de negocio en notebooks.

### 1.5 Exportacion sistematica de figuras

Las 12 visualizaciones del notebook 03 se guardan en `reports/figures/` con nombres descriptivos (`viz1_distribucion_precios.png`, etc.) y DPI=150. Se usa `plt.close(fig)` para liberar memoria.

### 1.6 Guardado en multiples formatos

El dataset procesado se guarda tanto en CSV como en Parquet, aprovechando la conservacion de tipos de Parquet y la portabilidad de CSV.

### 1.7 Deteccion robusta de PROJECT_ROOT

Los notebooks implementan multiples estrategias de fallback para detectar la raiz del proyecto, incluyendo busqueda por estructura de directorios y variable de entorno.

---

## 2. Areas de Mejora

### 2.1 Analisis de mecanismo de datos faltantes (MCAR/MAR/MNAR)

**Estado actual**: El notebook 01 genera un `null_report()` y visualiza con `missingno`, pero no analiza **por que** faltan los datos.

**Recomendacion**: Antes de imputar, determinar si los nulos son:
- **MCAR** (Missing Completely at Random): No hay patron, safe para imputar con media/mediana
- **MAR** (Missing at Random): El nulo depende de otra variable observada
- **MNAR** (Missing Not at Random): El nulo depende del valor que falta

```python
# Patron recomendado: analizar si los nulos correlacionan con otras variables
def analyze_missing_mechanism(df: pd.DataFrame, target_col: str, group_cols: list) -> None:
    """Analiza si los nulos de target_col estan asociados a group_cols."""
    df['_is_null'] = df[target_col].isnull().astype(int)
    for col in group_cols:
        if df[col].dtype in [np.float64, np.int64]:
            corr = df['_is_null'].corr(df[col])
            if abs(corr) > 0.1:
                print(f"  {col}: correlacion con nulos = {corr:.3f} (posible MAR)")
    df.drop(columns=['_is_null'], inplace=True)
```

**Prioridad**: Media

---

### 2.2 Unificar la deteccion de PROJECT_ROOT

**Estado actual**: Cada notebook implementa su propia funcion de deteccion (`_resolve_project_root()`, `_find_project_root()`). Hay 3 variantes distintas en los 4 notebooks.

**Recomendacion**: Centralizar en un modulo unico:

```python
# src/utils/paths.py (NUEVO)
from pathlib import Path
import os

def get_project_root() -> Path:
    """Resuelve la raiz del proyecto de forma robusta."""
    # Estrategia 1: subir desde cwd buscando marcadores
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        if (parent / 'src').exists() and (parent / 'data').exists():
            return parent
    # Estrategia 2: variable de entorno
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).exists():
        return Path(env_root)
    raise EnvironmentError("No se pudo detectar PROJECT_ROOT.")
```

Y en cada notebook simplemente:

```python
from src.utils.paths import get_project_root
PROJECT_ROOT = get_project_root()
```

**Prioridad**: Alta (reduce duplicacion y riesgo de divergencia)

---

### 2.3 Usar `pd.pipe()` para pipelines de transformaciones

**Estado actual**: Las transformaciones en notebook 02 se aplican secuencialmente:

```python
df = compute_review_features(df)
df = compute_time_features(df)
df = compute_price_features(df)
df = compute_platform_features(df)
```

**Recomendacion**: Usar `pd.pipe()` para un pipeline mas legible y encadenable:

```python
df = (
    df.pipe(compute_review_features)
      .pipe(compute_time_features)
      .pipe(compute_price_features)
      .pipe(compute_platform_features)
)
```

**Ventaja**: Mas legible, inmutable por diseño, facilita testing individual de cada paso.

**Prioridad**: Baja (mejora estetica, no funcional)

---

### 2.4 Profiling automatizado como complemento

**Estado actual**: El EDA se realiza manualmente con funciones propias, lo cual es excelente para entender los datos. Sin embargo, no se usa un profiler automatizado como primera pasada.

**Recomendacion**: Agregar `ydata-profiling` (antes pandas-profiling) como paso opcional de inspeccionar rapido:

```python
# Opcional: profiling automatizado como sanity check
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Steam Games - Quick Profile", minimal=True)
profile.to_file(FIGURES_DIR / "profile_report.html")
```

Esto complementa (no reemplaza) el analisis manual y puede revelar patrones no esperados.

**Prioridad**: Baja (complementario al flujo actual)

---

### 2.5 Analisis de multicolinealidad (VIF)

**Estado actual**: El notebook 03 calcula la matriz de correlacion de Pearson e identifica pares con `|r| > 0.3`. Se detectaron correlaciones altas:
- `positive` vs `peak_ccu`: r = 0.85
- `negative` vs `peak_ccu`: r = 0.83
- `negative` vs `positive`: r = 0.82

**Recomendacion**: Calcular el VIF (Variance Inflation Factor) para detectar multicolinealidad que la correlacion de pares no captura:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Calcula VIF para detectar multicolinealidad."""
    X = df[cols].dropna()
    vif_data = pd.DataFrame({
        'feature': cols,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(len(cols))]
    })
    return vif_data.sort_values('VIF', ascending=False)

# VIF > 10 indica multicolinealidad problematica
vif = compute_vif(df, numeric_cols)
print(vif[vif['VIF'] > 10])
```

Ademas, complementar con correlacion de **Spearman** (monotonica, no asume linealidad) para variables con distribuciones asimetricas como `price`, `positive`, `peak_ccu`.

**Prioridad**: Alta (impacta directamente la calidad de modelos en notebook futuro)

---

### 2.6 PerformanceWarning por fragmentacion del DataFrame

**Estado actual**: El notebook 03 genera un warning:

```
PerformanceWarning: DataFrame is highly fragmented. This is usually the result
of calling `frame.insert` many times. Consider joining all columns at once
using pd.concat(axis=1) instead.
```

**Causa**: Creacion iterativa de columnas one-hot y TF-IDF con asignacion individual.

**Recomendacion**: Agrupar las columnas nuevas y concatenarlas de una sola vez:

```python
# En lugar de:
for genre in top_genres:
    df[f'genre_{genre}'] = (df['genre_primary'] == genre).astype(int)

# Hacer:
genre_dummies = pd.DataFrame({
    f'genre_{genre.lower()}': (df['genre_primary'] == genre).astype(int)
    for genre in top_genres
})
df = pd.concat([df, genre_dummies], axis=1)
```

Despues de agregar muchas columnas, ejecutar `df = df.copy()` para desfragmentar.

**Prioridad**: Media (afecta rendimiento en datasets grandes)

---

### 2.7 Separacion de analisis univariado y bivariado

**Estado actual**: El notebook 03 mezcla visualizaciones univariadas (histogramas de precio) con bivariadas (precio vs reviews) y multivariadas (heatmap de correlacion) sin separacion clara.

**Recomendacion**: Organizar las visualizaciones por tipo de analisis con secciones markdown claras:

```
## Analisis Univariado
  - Viz 1: Distribucion de precios
  - Viz 2: Top generos
  - Viz 9: Metacritic scores

## Analisis Bivariado
  - Viz 6: Precio vs Reviews
  - Viz 10: Boxplots genero vs precio
  - Viz 7: Comparativa plataformas

## Analisis Multivariado
  - Viz 5: Matriz de correlacion
  - Viz 4: Reviews (scatter + pie + histograma)

## Analisis Temporal
  - Viz 3: Lanzamientos por año
  - Viz 11: Evolucion del precio
```

**Prioridad**: Baja (mejora organizativa)

---

### 2.8 Falta de pairplot para relaciones multivariadas

**Estado actual**: Se usa scatter plot individual y heatmap de correlacion, pero no hay un `pairplot` que muestre distribuciones y relaciones simultaneamente.

**Recomendacion** (segun documentacion de seaborn via Context7):

```python
# Seleccionar variables clave (max 5-6 para legibilidad)
key_vars = ['price', 'positive_ratio', 'log_total_reviews',
            'game_age_years', 'n_languages']

# Pairplot con hue por genero
sample = df[df['genre_primary'].isin(['Action', 'Indie', 'RPG'])].sample(2000, random_state=42)
sns.pairplot(
    sample[key_vars + ['genre_primary']],
    hue='genre_primary',
    diag_kind='kde',
    plot_kws={'alpha': 0.3, 's': 10},
    palette='husl'
)
plt.suptitle('Pairplot de Variables Clave por Genero', y=1.02)
plt.savefig(FIGURES_DIR / 'pairplot_variables_clave.png', dpi=150, bbox_inches='tight')
```

**Prioridad**: Media (agrega valor visual significativo con minimo esfuerzo)

---

### 2.9 Agregar ECDF como alternativa a histogramas

**Estado actual**: Se usan histogramas con KDE para todas las distribuciones.

**Recomendacion** (segun documentacion de seaborn via Context7): La funcion empirica de distribucion acumulada (`ecdfplot`) es mas informativa que un histograma para comparar distribuciones entre grupos, ya que no depende del tamano de bins:

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.ecdfplot(data=df[df['price'] > 0], x='price', hue='genre_primary',
             hue_order=top_5_genres, ax=ax)
ax.set_xlim(0, 60)
ax.set_title('ECDF de Precio por Genero')
```

**Prioridad**: Baja (complementario)

---

### 2.10 Tests unitarios para funciones de transformacion

**Estado actual**: Las validaciones estan dentro de los notebooks (assertions), pero no existen tests formales en `tests/`.

**Recomendacion**: Migrar las funciones de `compute_*` a `src/features/` y crear tests:

```python
# tests/test_features.py
import pytest
import pandas as pd
import numpy as np
from src.features.review_features import compute_review_features

def test_compute_review_features_preserves_rows():
    df = pd.DataFrame({'positive': [10, 0, 5], 'negative': [2, 0, 3]})
    result = compute_review_features(df)
    assert len(result) == len(df)

def test_positive_ratio_bounds():
    df = pd.DataFrame({'positive': [10, 0], 'negative': [2, 0]})
    result = compute_review_features(df)
    valid = result['positive_ratio'].dropna()
    assert valid.between(0, 1).all()
```

**Prioridad**: Alta (el proyecto lista `tests/` como pendiente en CLAUDE.md)

---

## 3. Recomendaciones por Notebook

### 3.1 Notebook 01: Limpieza y Exploracion

| # | Recomendacion | Impacto | Esfuerzo |
|---|---|---|---|
| 1 | Agregar analisis MCAR/MAR/MNAR antes de imputar | Alto | Medio |
| 2 | Usar `missingno.heatmap()` ademas de `matrix()` para ver correlaciones entre nulos | Medio | Bajo |
| 3 | Documentar la decision de eliminar `detailed_description` vs usar NLP en ella | Medio | Bajo |
| 4 | Agregar validacion de schema post-carga (dtypes esperados vs reales) | Medio | Medio |
| 5 | Considerar `pd.StringDtype()` en lugar de `object` para columnas de texto (pandas 2+) | Bajo | Bajo |

### 3.2 Notebook 02: Transformaciones y Features

| # | Recomendacion | Impacto | Esfuerzo |
|---|---|---|---|
| 1 | Migrar funciones `compute_*` a `src/features/` | Alto | Medio |
| 2 | Usar `pd.pipe()` para encadenar transformaciones | Bajo | Bajo |
| 3 | Agregar VIF despues de crear features escaladas | Alto | Bajo |
| 4 | Documentar el rationale de `n_features=15` en TF-IDF (no hardcodear) | Medio | Bajo |
| 5 | Considerar `TargetEncoder` de scikit-learn para developer/publisher (en lugar de frequency) | Medio | Medio |

### 3.3 Notebook 03: Visualizaciones e Insights

| # | Recomendacion | Impacto | Esfuerzo |
|---|---|---|---|
| 1 | Agregar pairplot de variables clave | Medio | Bajo |
| 2 | Usar ECDF para comparar distribuciones entre grupos | Medio | Bajo |
| 3 | Organizar visualizaciones por tipo (univariado/bivariado/multivariado) | Bajo | Bajo |
| 4 | Corregir PerformanceWarning de DataFrame fragmentado | Medio | Bajo |
| 5 | Agregar `violinplot` o `boxenplot` (seaborn) para distribuciones con muchos datos | Bajo | Bajo |

### 3.4 Notebook 04: Reporte Final

| # | Recomendacion | Impacto | Esfuerzo |
|---|---|---|---|
| 1 | Agregar intervalos de confianza a los KPIs (no solo valores puntuales) | Medio | Medio |
| 2 | Incluir seccion de limitaciones del dataset | Alto | Bajo |
| 3 | Agregar link a los notebooks fuente de cada hallazgo | Bajo | Bajo |
| 4 | Considerar exportar el reporte como HTML estatico con `nbconvert` | Medio | Bajo |

---

## 4. Resumen de Prioridades

### Alta Prioridad (implementar pronto)

1. **Unificar `get_project_root()`** en `src/utils/paths.py` - elimina 3 duplicados
2. **Calcular VIF** para detectar multicolinealidad antes de modelar
3. **Crear `tests/`** con tests unitarios para funciones de transformacion
4. **Migrar funciones `compute_*`** de notebook 02 a `src/features/`

### Media Prioridad (mejora iterativa)

5. Analisis de mecanismo de datos faltantes (MCAR/MAR/MNAR)
6. Corregir fragmentacion del DataFrame (PerformanceWarning)
7. Agregar pairplot de variables clave
8. Documentar decisiones de limpieza y parametros hardcodeados

### Baja Prioridad (nice-to-have)

9. Usar `pd.pipe()` para pipelines
10. Agregar ECDF como complemento a histogramas
11. Profiling automatizado con `ydata-profiling`
12. Reorganizar visualizaciones por tipo de analisis

---

## 5. Checklist de Accion

- [ ] Crear `src/utils/paths.py` con `get_project_root()` unificado
- [ ] Migrar `compute_review_features()`, `compute_time_features()`, `compute_price_features()`, `compute_platform_features()` a `src/features/steam_features.py`
- [ ] Crear `tests/test_steam_features.py` con tests basicos
- [ ] Agregar VIF al notebook 02 o 03 despues de la matriz de correlacion
- [ ] Corregir PerformanceWarning en notebook 03 (usar `pd.concat` en lugar de asignacion iterativa)
- [ ] Agregar seccion de limitaciones al notebook 04
- [ ] Agregar `ydata-profiling` a `requirements.txt` (opcional)
- [ ] Agregar `statsmodels` a `requirements.txt` (para VIF)

---

## 6. Referencias

- **pandas**: Documentacion oficial de manejo de missing data, `DataFrame.pipe()`, tipos de datos optimizados
- **seaborn**: `displot`, `histplot`, `ecdfplot`, `pairplot`, `heatmap`, `violinplot`, `boxenplot`
- **scikit-learn**: `StandardScaler`, `RobustScaler`, `TfidfVectorizer`, `TargetEncoder`
- **Mejores practicas EDA**: Checklist del proyecto en `docs/best_practices_EDA.md`
- **Context7 pandas docs**: Estrategias de imputacion, `fillna()`, `dropna()`, `interpolate()`
- **Context7 seaborn docs**: Distribution plots, categorical plots, matrix plots, pairplots

---

*Informe generado el 2026-02-13. Basado en el analisis de 4 notebooks del proyecto Steam Games con ~122,610 registros.*
