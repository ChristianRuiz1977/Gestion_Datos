# CLAUDE.md — Gestion_Datos

Contexto del proyecto para Claude Code. Este archivo guía el comportamiento del asistente en este repositorio.

---

## Descripción del Proyecto

Workspace de Maestría en Ciencia de Datos orientado a análisis y gestión de datos. Es un proyecto **template estructurado** que sigue las convenciones estándar de proyectos de data science, cubriendo el ciclo completo: ingesta de datos → EDA → features → modelos → reportes.

**Lenguaje principal**: Python 3.9+

---

## Estructura del Proyecto

```
Gestion_Datos/
├── data/
│   ├── raw/           # Datos originales — NUNCA modificar directamente
│   ├── processed/     # Datos limpios y transformados
│   └── external/      # Fuentes de datos externas o de terceros
├── notebooks/
│   ├── exploratory/   # Análisis exploratorio (EDA)
│   ├── reports/       # Notebooks pulidos para presentación
│   └── tutorials/     # Material de aprendizaje
├── src/               # Código Python reutilizable (paquete)
│   ├── data/
│   │   └── loader.py  # Carga multi-formato: CSV, XLSX, JSON
│   ├── features/      # Ingeniería de features (pendiente de implementar)
│   ├── models/        # Entrenamiento y evaluación de modelos (pendiente)
│   └── visualization/
│       └── plots.py   # Utilidades de visualización (histogramas, heatmaps)
├── reports/
│   ├── figures/       # Plots exportados
│   └── tables/        # Tablas y métricas de análisis
├── venv/              # Entorno virtual (no modificar, no commitear)
├── requirements.txt
└── README.md
```

---

## Stack Tecnológico

| Categoría | Librerías |
|---|---|
| Datos | pandas ≥1.3, numpy ≥1.21, scipy ≥1.7 |
| ML | scikit-learn ≥1.0, xgboost ≥1.5 |
| Visualización | matplotlib ≥3.4, seaborn ≥0.11, plotly ≥5.0 |
| Notebooks | JupyterLab ≥3.0, IPython ≥7.0 |
| Calidad de código | black ≥21.0, flake8 ≥3.9 |
| Testing | pytest ≥6.2 |
| Config | python-dotenv ≥0.19 |

---

## Flujo de Trabajo Principal

```
data/raw/
  └─ src/data/loader.py ──► data/processed/
                                └─ src/features/  (EDA + transformaciones)
                                      └─ src/models/  (entrenamiento)
                                            └─ reports/figures/ + reports/tables/
```

Usar notebooks en `notebooks/exploratory/` para iterar y prototipar.
Mover código estable a módulos dentro de `src/`.

---

## Comandos Frecuentes

```bash
# Entorno
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Desarrollo
jupyter lab                    # Iniciar entorno de notebooks
pytest tests/                  # Ejecutar tests
black src/ notebooks/          # Formatear código
flake8 src/                    # Verificar estilo
```

---

## Convenciones y Reglas

### Datos
- `data/raw/` es **inmutable** — los archivos originales nunca se sobrescriben.
- Nombres de archivo con fecha: `raw_ventas_2024-01-15.csv`, `processed_clientes_2024-01-15.csv`.
- No commitear datos sensibles ni archivos `.env`. Ya están en `.gitignore`.

### Código Python
- Estilo **PEP 8** enforced con `black` y `flake8`.
- Toda función pública debe tener **docstring**.
- Usar **type hints** en firmas de funciones (e.g., `-> pd.DataFrame`).
- Pathlib para manejo de rutas, no strings hardcodeados.

### Notebooks
- Numerar con prefijo: `01_eda_inicial.ipynb`, `02_limpieza.ipynb`.
- Reiniciar kernel y ejecutar completo antes de commitear.
- Solo commitear notebooks con output limpio (o sin output).

### Módulos `src/`
- Código que se reutiliza entre notebooks va en `src/`.
- No escribir lógica de negocio dentro de notebooks; usarlos solo para exploración.
- `src/features/` y `src/models/` están como stubs — implementar conforme avance el proyecto.

### Visualizaciones
- Usar `src/visualization/plots.py` como base.
- Exportar figuras a `reports/figures/` en formato `.png` o `.svg`.
- Estilo por defecto: `seaborn-v0_8-darkgrid`, paleta `husl`.

---

## Archivos Clave

| Archivo | Descripción |
|---|---|
| [src/data/loader.py](src/data/loader.py) | `load_data(filepath)` — carga CSV/XLSX/JSON a DataFrame |
| [src/visualization/plots.py](src/visualization/plots.py) | `plot_distribution()`, `plot_correlation()`, `set_style()` |
| [requirements.txt](requirements.txt) | Dependencias del proyecto |
| [data/README.md](data/README.md) | Convenciones y guía de la carpeta de datos |
| [reports/README.md](reports/README.md) | Estándares para reportes y figuras |
| [.github/copilot-instructions.md](.github/copilot-instructions.md) | Instrucciones adicionales para asistentes de IA |

---

## Lo que NO Hacer

- No modificar archivos en `data/raw/` — son la fuente de verdad.
- No commitear `venv/`, `__pycache__/`, `.ipynb_checkpoints/`, ni archivos `.env`.
- No instalar paquetes globalmente; siempre dentro del entorno virtual activado.
- No crear archivos de código fuera de `src/` sin justificación (evitar dispersión).
- No usar `print()` como único mecanismo de debug en módulos de `src/`; preferir `logging`.
- No hardcodear rutas absolutas — usar `pathlib.Path` relativo a la raíz del proyecto.

---

## Estado Actual del Proyecto

| Módulo | Estado |
|---|---|
| `src/data/loader.py` | Implementado |
| `src/visualization/plots.py` | Implementado |
| `src/features/` | Stub (pendiente) |
| `src/models/` | Stub (pendiente) |
| `notebooks/` | Estructura creada, sin notebooks aún |
| `tests/` | No existe aún — crear junto con nuevos módulos |

---

## Notas para Claude Code

- Al añadir funciones a `src/`, seguir el estilo existente: docstrings, type hints, pathlib.
- Antes de crear un nuevo archivo, verificar si existe un módulo relevante en `src/`.
- Si se pide análisis de datos, asumir que los archivos están en `data/raw/` o `data/processed/`.
- Para visualizaciones, extender `src/visualization/plots.py` en lugar de código ad-hoc.
- Al generar notebooks, usar la carpeta `notebooks/exploratory/` con el prefijo numérico.
- Reportar cualquier dependencia nueva que se necesite agregar a `requirements.txt`.
