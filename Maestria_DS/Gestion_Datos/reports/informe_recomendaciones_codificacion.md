# Informe de Recomendaciones de Codificación

**Proyecto:** Gestion_Datos — Steam Games Analytics Pipeline
**Notebooks analizados:**
- `notebooks/exploratory/01_limpieza_exploracion_steam.ipynb`
- `notebooks/exploratory/02_transformaciones_features_steam.ipynb`
- `notebooks/exploratory/03_visualizaciones_insights_steam.ipynb`
- `notebooks/reports/04_reporte_final_steam.ipynb`

**Módulos de referencia:** `src/data/loader.py`, `src/visualization/plots.py`
**Fecha:** 2026-02-12

---

## Resumen Ejecutivo

El análisis cubre dos dimensiones complementarias: patrones de **ETL** (extracción, transformación y carga) y patrones de **visualización**. En total se identificaron **30 hallazgos** distribuidos en 4 categorías de severidad. Los problemas más críticos afectan la reproducibilidad del pipeline, la ausencia de tests unitarios, y la no utilización efectiva de los módulos centralizados del proyecto (`src/`).

| Severidad | ETL | Visualización | Total |
|-----------|-----|---------------|-------|
| Alta      | 4   | 3             | **7** |
| Media     | 6   | 5             | **11**|
| Baja      | 6   | 6             | **12**|

---

## Parte I — Patrones ETL

### EXT-01 · Fallback a ruta absoluta hardcodeada `[Media]`

**Afecta:** NB01, NB02, NB04 — celda de Setup en cada uno.

Los tres notebooks implementan una detección de `PROJECT_ROOT` con un fallback final a una ruta absoluta local:

```python
# NB02 — ejemplo representativo
return Path(r'C:\Users\Christian Ruiz\Maestria_DS\Gestion_Datos')
```

Aunque la intención de detectar el root dinámicamente es correcta, el fallback hardcodeado rompe la portabilidad si el proyecto se ejecuta en otro equipo, contenedor o entorno de CI. Falla silenciosamente sin indicar la causa.

**Recomendación:** Eliminar el fallback y lanzar un error explícito si ninguna heurística funciona. Complementariamente, usar `python-dotenv` (ya incluido en `requirements.txt`) para definir `PROJECT_ROOT` en un archivo `.env`:

```
# .env (no commitear)
PROJECT_ROOT=C:\Users\Christian Ruiz\Maestria_DS\Gestion_Datos
```

---

### EXT-02 · Uso de `glob.glob` en lugar de `pathlib.glob` `[Baja]`

**Afecta:** NB04 — celda de carga de features.

El NB04 importa `glob` del módulo estándar y construye el patrón como string, mientras que el NB02 usa correctamente `Path.glob()`. Genera inconsistencia con la convención de `pathlib` establecida en el `CLAUDE.md`.

**Recomendación:** Unificar en `pathlib.Path.glob()` y mover la función `find_latest_processed_file` (definida en NB02) a `src/data/loader.py` para que sea reutilizable desde cualquier notebook.

---

### EXT-03 · Ausencia de validación de integridad post-carga `[Media]`

**Afecta:** NB04 — carga del CSV de features.

Tras `pd.read_csv()` solo se imprime el shape. Si el archivo está vacío, corrupto o fue generado por una ejecución parcial del NB02, el pipeline de reporte continúa con datos inválidos.

**Recomendación:** Agregar assertions mínimas de integridad post-carga que fallen con mensajes claros:

```python
assert df.shape[0] >= 10_000, f"Solo {df.shape[0]} filas — dataset incompleto."
assert 'price_tier' in df.columns, "Columna 'price_tier' ausente — re-ejecutar NB02."
```

---

### TRF-01 · Funciones de transformación reutilizables definidas en los notebooks `[Alta]`

**Afecta:** NB01 (`extract_list_names`, `extract_tags_top`, `_parse_languages`, `parse_owners_midpoint`).

Estas funciones implementan lógica de transformación compleja pero viven dentro del notebook. El NB02 consume columnas derivadas de ellas sin acceso a su implementación original. Es una violación directa del principio del `CLAUDE.md`: *"Código que se reutiliza entre notebooks va en `src/`"*.

**Recomendación:** Mover todas las funciones de parseo de campos Steam a `src/data/steam_transforms.py` e importarlas desde los notebooks.

---

### TRF-02 · `df.apply` fila a fila para clasificación de etiquetas de review `[Media]`

**Afecta:** NB02 — función `compute_review_features`.

La clasificación de etiquetas Steam (`Overwhelmingly Positive`, `Very Positive`, etc.) usa `df.apply(steam_review_label, axis=1)` sobre ~122,000 filas. El mismo notebook vectoriza correctamente la clasificación de plataformas, lo que evidencia inconsistencia de criterio.

**Recomendación:** Reemplazar por `np.select()` con condiciones vectorizadas, que es entre 10× y 50× más rápido para este tipo de clasificación ordinal.

---

### TRF-03 · `platform_combo` usa `apply` sobre columnas booleanas `[Baja-Media]`

**Afecta:** NB02 — función `compute_platform_features`.

La construcción del string de combinación de plataformas (`Win+Mac+Lin`) usa `apply` fila a fila sobre columnas booleanas, siendo completamente vectorizable.

**Recomendación:** Construir el string mediante operaciones de máscara booleana y `str.join` vectorizado, eliminando la iteración fila a fila.

---

### TRF-04 · Ausencia de validaciones post-transformación `[Alta]`

**Afecta:** NB02 — funciones `compute_review_features`, `compute_time_features`, `compute_price_features`, `compute_platform_features`.

Ninguna función valida que su output cumpla invariantes esperadas. Por ejemplo, `positive_ratio` podría tener valores fuera de `[0.0, 1.0]` o `game_age_years` valores negativos sin que el pipeline lo detecte.

**Recomendación:** Agregar un bloque de `assert` al final de cada función de transformación que verifique: rangos de valores, tipos de datos, y que el número de filas de entrada y salida sea idéntico.

---

### TRF-05 · Función auxiliar definida dentro de una celda de limpieza `[Baja]`

**Afecta:** NB01 — `_parse_languages` definida dentro de la celda de tratamiento de nulos.

Mezcla definición de lógica reutilizable con su aplicación, dificultando la lectura del flujo ETL.

**Recomendación:** Consolidar junto con las demás funciones de parseo en `src/data/steam_transforms.py` (ver TRF-01).

---

### TRF-06 · `parse_owners_midpoint` sin tests para edge cases `[Media]`

**Afecta:** NB02 — parseo del campo `estimated_owners`.

La función retorna `0.0` silenciosamente ante inputs inválidos. No tiene cobertura para casos como `"0 - 0"`, valores con comas (`"1,000,000 - 2,000,000"`), o strings vacíos.

**Recomendación:** Crear `tests/test_steam_transforms.py` con tests parametrizados que cubran estos casos. Es especialmente crítico porque fallas silenciosas aquí producen ceros en una variable de negocio clave.

---

### TRF-07 · Reconstrucción duplicada e inconsistente de `price_tier` en NB04 `[Media]`

**Afecta:** NB04 — celdas 14 y 16.

El NB04 reconstruye `price_tier` dos veces con etiquetas distintas (`'Gratis'`, `'$0.01-$5'`...) que no coinciden con las del NB02 (que usa `Free / Low / Mid / Standard / Premium / AAA`). Rompe la consistencia de la variable a lo largo del pipeline.

**Recomendación:** Consumir `price_tier` directamente desde el CSV de features generado por el NB02. Si la columna no está presente, fallar explícitamente en lugar de reconstruirla con lógica divergente.

---

### LOAD-01 · Tablas analíticas del reporte no persisten en `reports/tables/` `[Baja-Media]`

**Afecta:** NB04.

El reporte genera tablas de KPIs, estadísticas por género y por era, pero no guarda ninguna en `reports/tables/`. El directorio existe pero permanece vacío. Los resultados analíticos no son reproducibles sin re-ejecutar el notebook completo.

**Recomendación:** Exportar cada tabla de análisis a `reports/tables/` con la convención de nombres con fecha del proyecto (`steam_kpis_2026-02-12.csv`).

---

### LOAD-02 · Conversión redundante de columnas `object` antes del guardado CSV `[Baja]`

**Afecta:** NB01 — celda de guardado.

El bucle `apply(lambda x: str(x))` sobre todas las columnas `object` es innecesario: `pandas.to_csv` serializa automáticamente cualquier tipo Python a string, y Parquet preserva los tipos nativamente.

**Recomendación:** Eliminar el bucle de conversión y guardar directamente. Priorizar Parquet como formato de intercambio entre notebooks.

---

### QA-01 · Ausencia total de tests unitarios `[Alta]`

**Afecta:** NB01, NB02 — aproximadamente 15 funciones de transformación en total.

El directorio `tests/` no existe. Ninguna de las funciones de transformación tiene cobertura de tests. Las funciones con retorno silencioso ante errores (`parse_owners_midpoint`, `extract_list_names`) son especialmente vulnerables a regresiones no detectadas.

**Recomendación:** Crear `tests/test_etl_transform.py` con al menos un test por función pública. La prioridad son funciones que retornan `0.0` o `''` como valor de error en lugar de lanzar excepciones.

---

### QA-02 · `select_dtypes('object')` genera `Pandas4Warning` activa `[Baja]`

**Afecta:** NB01 — estadísticas descriptivas de texto.

La llamada `df.select_dtypes('object')` produce una advertencia de deprecación visible en la salida del notebook ejecutado, indicando incompatibilidad futura con Pandas 3+.

**Recomendación:** Actualizar a `select_dtypes(include=['object', 'str'])` para incluir explícitamente ambos tipos y eliminar la advertencia.

---

### QA-03 · Clasificación de tipo de columna basada en primeros N valores `[Baja-Media]`

**Afecta:** NB01 — función `classify_columns`.

La función inspecciona solo los primeros 20 valores no nulos para inferir el tipo de una columna. Con 122,000 filas y datos mixtos, esto puede producir clasificaciones incorrectas.

**Recomendación:** Aumentar la muestra a 100 valores y usar votación mayoritaria del tipo dominante en lugar de tomar solo el primer valor.

---

### QA-04 · `warnings.filterwarnings('ignore')` global en el reporte final `[Baja]`

**Afecta:** NB04 — celda de setup.

Suprime todas las advertencias de Python, ocultando los `FutureWarning` y `DeprecationWarning` identificados en otros hallazgos (A8, A9, QA-02).

**Recomendación:** Reemplazar por supresiones específicas y acotadas al contexto donde son necesarias, nunca globalmente.

---

## Parte II — Patrones de Visualización

### A1 · Importaciones de `plots.py` sin uso efectivo `[Alta]`

**Afecta:** NB01, NB03 — celda de setup en ambos.

Ambos notebooks importan `plot_distribution` y `plot_correlation` desde `src.visualization.plots`, pero nunca las invocan. Todo el código de visualización se escribe ad-hoc en cada celda, duplicando lógica que ya existe en el módulo centralizado.

**Causa raíz:** Las funciones actuales de `plots.py` no son reutilizables en subplots porque no aceptan un eje `ax` externo ni soportan exportación (ver A14). Esto impide adoptarlas para gráficos reales.

---

### A2 · `sns.set_style()` directo bypasea `set_style()` del módulo `[Media]`

**Afecta:** NB04 — celda de distribuciones (celda 10).

Se llama directamente a `sns.set_style('darkgrid')` en una celda intermedia, sobrescribiendo el estilo canónico del proyecto (`seaborn-v0_8-darkgrid` + paleta `husl`) que fue configurado por `set_style()` en el setup.

**Recomendación:** Eliminar todas las llamadas directas a `sns.set_style()` en celdas individuales. El estilo solo debe configurarse una vez, desde `set_style()` del módulo.

---

### A3 · Paletas de colores hexadecimales hardcodeadas `[Media]`

**Afecta:** NB03 — celdas viz3, viz7, viz9, viz12, viz14.

Se repiten definiciones de paletas con valores hexadecimales fijos en al menos 5 celdas distintas, sin relación con la paleta `husl` del proyecto.

**Recomendación:** Centralizar las paletas nombradas semánticamente en `src/visualization/plots.py` o en un módulo `src/styles/constants.py`, y referenciarlas desde los notebooks.

---

### A4 · `rcParams` sobreescritos manualmente después de `set_style()` `[Baja]`

**Afecta:** NB03 — celda de setup.

Después de llamar a `set_style()`, el notebook sobreescribe `figure.dpi` y `font.family` manualmente. Estos parámetros deberían ser parte de `set_style()`.

**Recomendación:** Incorporar `figure.dpi`, `font.family` y otros parámetros de calidad directamente en la firma de `set_style()` con valores por defecto documentados.

---

### A5 · `plt.show()` en lugar de `plt.close(fig)` en celda de exportación `[Baja]`

**Afecta:** NB01 — celda del heatmap de nulos.

Es la única celda en todo el proyecto que usa `plt.show()` en un contexto de exportación, inconsistente con el patrón `plt.close(fig)` usado en el resto del proyecto.

**Recomendación:** Reemplazar por `plt.close(fig)` para consistencia y para evitar doble renderizado en ejecuciones con `nbconvert`.

---

### A6 · Ausencia de exportación SVG en el reporte final `[Baja]`

**Afecta:** NB04 — celdas de figuras del reporte.

El `CLAUDE.md` especifica exportar en `.png` o `.svg`. Las figuras del reporte solo se exportan en `.png`. Para un reporte orientado a presentación ejecutiva, SVG es preferible por ser vectorial.

**Recomendación:** Exportar las figuras del reporte en ambos formatos, o al menos en SVG.

---

### A7 · Lógica de heatmap de correlación duplicada entre notebooks `[Alta]`

**Afecta:** NB03 (viz5) y NB04 (reporte de correlaciones).

Ambos notebooks construyen heatmaps de correlación con lógica casi idéntica (`sns.heatmap`, máscara triangular, parámetros de `annot`), pero con pequeñas variaciones que dificultan el mantenimiento.

**Recomendación:** Extender `plot_correlation()` en `plots.py` para soportar máscara triangular, paleta configurable, `figsize`, `save_path` y parámetro `ax`. Esta función debe ser la única fuente de verdad para este tipo de gráfico.

---

### A8 · `FutureWarning` de seaborn: `palette` sin `hue` `[Alta]`

**Afecta:** NB03 — celda viz10 (boxplot y violinplot).

Las llamadas a `sns.boxplot()` y `sns.violinplot()` usan `palette` sin `hue`, generando `FutureWarning` visible en la salida. Este patrón será un error en seaborn v0.14.

**Recomendación:** Asignar `hue` a la misma variable que `x` y agregar `legend=False`:

```python
sns.boxplot(..., hue='genre_primary', palette='husl', legend=False)
```

---

### A9 · `DeprecationWarning` de matplotlib: parámetro `labels` obsoleto `[Media]`

**Afecta:** NB03 — celda viz7 (comparativa plataformas).

La llamada a `axes[1].boxplot()` usa el parámetro `labels`, renombrado a `tick_labels` en Matplotlib 3.9. El soporte para el nombre anterior se eliminará en 3.11.

**Recomendación:** Reemplazar `labels=` por `tick_labels=` en la llamada a `boxplot()`.

---

### A10 · Mutación del DataFrame principal en celdas de visualización `[Media]`

**Afecta:** NB03 (viz12) y NB04 (celda 16).

Las celdas de visualización modifican `df` directamente con nuevas columnas (`price_category`, `log_total_reviews`, `price_tier`), generando `PerformanceWarning` en el NB03 y efectos secundarios entre celdas.

**Recomendación:** Crear copias locales para transformaciones temporales de visualización (`df_viz = df[cols].copy()`) y nunca mutar el DataFrame principal dentro de celdas de graficado.

---

### A11 · Ausencia de visualizaciones interactivas con Plotly en el reporte final `[Media]`

**Afecta:** NB04.

Todas las figuras del reporte son estáticas. El stack del proyecto incluye Plotly y el `CLAUDE.md` especifica usarlo donde corresponda. Un scatter plot de precio vs. reviews o el heatmap de correlaciones son candidatos naturales para interactividad en un reporte ejecutivo.

**Recomendación:** Reemplazar al menos los gráficos de scatter y el heatmap en NB04 por equivalentes en Plotly, exportando como `.html` interactivo a `reports/figures/`.

---

### A12 · Sin anotaciones informativas en gráficos del reporte `[Baja]`

**Afecta:** NB04 — celdas 10, 12, 14.

Los gráficos del reporte final carecen de anotaciones contextuales que los distingan de un gráfico exploratorio. No se señalan umbrales relevantes, valores atípicos notables, ni conclusiones de negocio.

**Recomendación:** Agregar anotaciones (`ax.annotate()`) en al menos los gráficos más importantes del reporte, alineadas con las conclusiones del análisis.

---

### A13 · `figsize` hardcodeados sin constantes estándar `[Baja]`

**Afecta:** NB01, NB03, NB04 — de forma generalizada.

Todos los tamaños de figura están hardcodeados por celda (`(16, 6)`, `(14, 10)`, `(18, 6)`, etc.) sin una convención documentada.

**Recomendación:** Definir tamaños estándar en `src/visualization/plots.py` o en un módulo de constantes:

```python
FIGSIZE_SINGLE    = (10, 6)    # Un panel
FIGSIZE_WIDE      = (14, 6)    # Dos paneles lado a lado
FIGSIZE_TALL      = (12, 10)   # Heatmaps y figuras cuadradas
FIGSIZE_DASHBOARD = (16, 12)   # Multi-panel (2×2 o 2×3)
```

---

### A14 · `plots.py` no es reutilizable en subplots `[Alta]`

**Afecta:** `src/visualization/plots.py` — causa raíz de A1.

Las funciones del módulo centralizado tienen limitaciones estructurales que impiden su adopción real:

| Limitación | Descripción |
|------------|-------------|
| `plot_distribution` llama `plt.show()` internamente | Impide usarla en subplots o controlar el momento de renderizado |
| Ninguna función acepta `ax` externo | No integrable en un `fig, axes = plt.subplots(...)` |
| Ninguna función tiene parámetro `save_path` | No soporta exportación desde la función |
| `plot_correlation` no soporta máscara triangular | Buena práctica que debe estar en el módulo |
| Ninguna función retorna la figura | Impide encadenar operaciones |
| `figsize` hardcodeado en las funciones | `(12, 6)` y `(12, 10)` fijos, no parametrizables |

**Recomendación:** Refactorizar las funciones de `plots.py` para que acepten `ax`, `save_path` y `figsize` como parámetros opcionales, y retornen la figura creada. Este cambio desbloquea todos los demás hallazgos relacionados con el módulo.

---

## Tabla Consolidada de Hallazgos

| ID | Área | Severidad | Notebook(s) | Anti-patrón |
|----|------|-----------|-------------|-------------|
| EXT-01 | Extract | Media | 01, 02, 04 | Ruta absoluta hardcodeada como fallback de `PROJECT_ROOT` |
| EXT-02 | Extract | Baja | 04 | `glob.glob` en lugar de `pathlib.glob` |
| EXT-03 | Extract | Media | 04 | Sin validación de integridad post-carga |
| TRF-01 | Transform | **Alta** | 01 | Funciones reutilizables definidas en el notebook en lugar de `src/` |
| TRF-02 | Transform | Media | 02 | `df.apply` fila a fila para clasificación vectorizable |
| TRF-03 | Transform | Baja-Media | 02 | `platform_combo` con `apply` sobre columnas booleanas |
| TRF-04 | Transform | **Alta** | 02 | Sin validaciones post-transformación en feature engineering |
| TRF-05 | Transform | Baja | 01 | Función auxiliar definida dentro de celda de limpieza |
| TRF-06 | Transform | Media | 02 | `parse_owners_midpoint` sin tests para edge cases |
| TRF-07 | Transform | Media | 04 | Reconstrucción duplicada e inconsistente de `price_tier` |
| LOAD-01 | Load | Baja-Media | 04 | Tablas del reporte no persisten en `reports/tables/` |
| LOAD-02 | Load | Baja | 01 | Conversión redundante de columnas antes del guardado CSV |
| QA-01 | Calidad | **Alta** | 01, 02 | Ausencia total de tests unitarios (~15 funciones) |
| QA-02 | Calidad | Baja | 01 | `select_dtypes('object')` genera `Pandas4Warning` |
| QA-03 | Calidad | Baja-Media | 01 | Clasificación de tipo de columna basada en primeros N valores |
| QA-04 | Calidad | Baja | 04 | `warnings.filterwarnings('ignore')` global |
| A1 | Visualización | **Alta** | 01, 03 | Importaciones de `plots.py` sin uso efectivo |
| A2 | Visualización | Media | 04 | `sns.set_style()` directo bypasea el módulo centralizado |
| A3 | Visualización | Media | 03 | Paletas hexadecimales hardcodeadas, fuera de convención `husl` |
| A4 | Visualización | Baja | 03 | `rcParams` sobreescritos manualmente post `set_style()` |
| A5 | Visualización | Baja | 01 | `plt.show()` en lugar de `plt.close(fig)` en exportación |
| A6 | Visualización | Baja | 04 | Sin exportación SVG en el reporte final |
| A7 | Visualización | **Alta** | 03, 04 | Lógica de heatmap duplicada entre notebooks |
| A8 | Visualización | **Alta** | 03 | `FutureWarning`: `palette` sin `hue` (error en seaborn v0.14) |
| A9 | Visualización | Media | 03 | `DeprecationWarning`: parámetro `labels` obsoleto en matplotlib 3.9 |
| A10 | Visualización | Media | 03, 04 | Mutación del DataFrame principal en celdas de visualización |
| A11 | Visualización | Media | 04 | Sin visualizaciones interactivas Plotly en el reporte final |
| A12 | Visualización | Baja | 04 | Sin anotaciones informativas en gráficos del reporte |
| A13 | Visualización | Baja | 01, 03, 04 | `figsize` hardcodeados sin constantes estándar |
| A14 | Visualización | **Alta** | `plots.py` | Módulo centralizado no acepta `ax`, `save_path` ni retorna figuras |

---

## Plan de Acción Priorizado

### Prioridad 1 — Bloquean reproducibilidad o corregirán errores en producción

| ID | Acción |
|----|--------|
| A14 | Refactorizar `plots.py`: añadir parámetros `ax`, `save_path`, `figsize`; retornar figura |
| A8 | Corregir `FutureWarning` de seaborn (`palette` sin `hue`) — error en v0.14 |
| A9 | Corregir `DeprecationWarning` de matplotlib (`labels` → `tick_labels`) |
| EXT-01 | Eliminar rutas absolutas hardcodeadas; usar `.env` con `python-dotenv` |
| TRF-01 | Migrar funciones de transformación Steam a `src/data/steam_transforms.py` |
| TRF-04 | Agregar validaciones post-transformación en funciones de feature engineering |
| QA-01 | Crear `tests/test_steam_transforms.py` con cobertura mínima de funciones críticas |

### Prioridad 2 — Afectan consistencia del proyecto

| ID | Acción |
|----|--------|
| A1 | Adoptar las funciones de `plots.py` en los notebooks (requiere completar A14) |
| A2, A3 | Centralizar estilo y paletas; eliminar llamadas directas a seaborn en notebooks |
| A7 | Extraer lógica del heatmap de correlación a función en `plots.py` |
| TRF-07 | Consumir `price_tier` del CSV de features; no reconstruir en NB04 |
| EXT-03 | Agregar assertions de integridad post-carga en NB04 |
| TRF-06 | Agregar tests para `parse_owners_midpoint` |

### Prioridad 3 — Mejoras de calidad y mantenibilidad

| ID | Acción |
|----|--------|
| TRF-02, TRF-03 | Vectorizar transformaciones que usan `df.apply` fila a fila |
| A10 | Usar copias locales para transformaciones temporales de visualización |
| A11 | Agregar gráficos Plotly interactivos en el reporte final |
| LOAD-01 | Exportar tablas analíticas a `reports/tables/` con nomenclatura de fecha |
| A13 | Definir constantes de `figsize` estándar en `plots.py` |
| QA-02 | Actualizar `select_dtypes` para compatibilidad con Pandas 3+ |
| QA-04 | Reemplazar `warnings.filterwarnings('ignore')` global por supresiones específicas |
| A4, A5, A6, A12 | Pulir detalles de estilo, exportación SVG y anotaciones en el reporte |
