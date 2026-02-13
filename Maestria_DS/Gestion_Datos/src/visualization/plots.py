"""Utilidades de visualización para el proyecto Steam Games Analytics.

Módulo centralizado de visualización. Todas las funciones:
- Aceptan un eje matplotlib externo (``ax``) para integración en subplots.
- Soportan exportación directa vía ``save_path``.
- Retornan la figura creada para permitir encadenamiento.
- Nunca llaman ``plt.show()`` internamente.

Uso típico en un notebook::

    from src.visualization.plots import set_style, plot_distribution, plot_correlation

    set_style()

    fig = plot_distribution(df['price'], title='Distribución de Precios',
                             save_path=FIGURES_DIR / 'dist_precio.png')
    plt.close(fig)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Constantes de tamaños estándar de figura
# ---------------------------------------------------------------------------
FIGSIZE_SINGLE: tuple[int, int] = (10, 6)     # Un panel
FIGSIZE_WIDE: tuple[int, int] = (14, 6)       # Dos paneles lado a lado
FIGSIZE_TALL: tuple[int, int] = (12, 10)      # Heatmaps y figuras cuadradas
FIGSIZE_DASHBOARD: tuple[int, int] = (16, 12) # Multi-panel (2×2 o 2×3)


def set_style(
    style: str = 'seaborn-v0_8-darkgrid',
    palette: str = 'husl',
    dpi: int = 120,
    font_family: str = 'DejaVu Sans',
) -> None:
    """Configura el estilo visual canónico del proyecto.

    Centraliza toda la configuración de matplotlib y seaborn en un único
    punto de entrada. Llamar esta función una vez al inicio de cada notebook
    es suficiente; no sobreescribir estilos en celdas individuales.

    Args:
        style: Nombre del estilo matplotlib/seaborn. Por defecto el estilo
               canónico del proyecto ('seaborn-v0_8-darkgrid').
        palette: Paleta de color seaborn. Por defecto 'husl'.
        dpi: Resolución de pantalla para figuras renderizadas. La resolución
             de exportación se controla con el parámetro ``dpi`` de
             ``fig.savefig()``. Por defecto 120.
        font_family: Familia tipográfica por defecto. Por defecto 'DejaVu Sans'.
    """
    plt.style.use(style)
    sns.set_palette(palette)
    plt.rcParams.update({
        'figure.dpi': dpi,
        'font.family': font_family,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
    })


def plot_distribution(
    data: pd.Series,
    title: str = None,
    figsize: tuple[int, int] = FIGSIZE_SINGLE,
    ax: plt.Axes = None,
    save_path: Path | str = None,
    **kwargs,
) -> plt.Figure:
    """Grafica la distribución de una Serie de pandas (histograma + KDE opcional).

    Args:
        data: Serie de pandas a graficar.
        title: Título del gráfico. Opcional.
        figsize: Tamaño de la figura como (ancho, alto) en pulgadas.
                 Ignorado si se provee ``ax``. Por defecto FIGSIZE_SINGLE.
        ax: Eje matplotlib externo. Si se provee, la función dibuja sobre él
            y no crea una nueva figura. Útil para subplots.
        save_path: Ruta de exportación (Path o str). Si es None, no exporta.
                   Exporta con ``dpi=150, bbox_inches='tight'``.
        **kwargs: Argumentos adicionales para ``seaborn.histplot``.

    Returns:
        La figura matplotlib creada (o la figura del eje externo si se
        proveyó ``ax``).

    Examples:
        # En notebook standalone
        fig = plot_distribution(df['price'], title='Precios',
                                save_path=FIGURES_DIR / 'dist_precio.png')
        plt.close(fig)

        # En subplot
        fig, axes = plt.subplots(1, 2)
        plot_distribution(df['price'], ax=axes[0])
        plot_distribution(df['positive_ratio'], ax=axes[1])
        fig.savefig(FIGURES_DIR / 'comparativa.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.histplot(data, ax=ax, **kwargs)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    ax.get_figure().tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_correlation(
    data: pd.DataFrame,
    title: str = 'Correlation Matrix',
    cmap: str = 'RdBu_r',
    mask_upper: bool = True,
    figsize: tuple[int, int] = FIGSIZE_TALL,
    ax: plt.Axes = None,
    save_path: Path | str = None,
) -> plt.Figure:
    """Grafica una matriz de correlación como heatmap anotado.

    Args:
        data: DataFrame con columnas numéricas. La correlación se calcula
              internamente con ``data.corr()``.
        title: Título del gráfico.
        cmap: Colormap de seaborn/matplotlib. Por defecto 'RdBu_r'
              (rojo-blanco-azul, centrado en cero).
        mask_upper: Si True, oculta el triángulo superior para evitar
                    duplicar información. Por defecto True.
        figsize: Tamaño de la figura. Ignorado si se provee ``ax``.
                 Por defecto FIGSIZE_TALL.
        ax: Eje matplotlib externo. Si se provee, dibuja sobre él.
        save_path: Ruta de exportación. Si es None, no exporta.

    Returns:
        La figura matplotlib creada.

    Examples:
        fig = plot_correlation(df[numeric_cols], title='Correlaciones',
                               save_path=FIGURES_DIR / 'corr.png')
        plt.close(fig)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    corr_matrix = data.corr()

    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        mask=mask,
        annot_kws={'size': 9},
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
