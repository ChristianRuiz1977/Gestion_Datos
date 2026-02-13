"""Feature engineering para el dataset de Steam Games.

Funciones de transformación que crean variables derivadas a partir del
dataset limpio.  Cada función recibe un DataFrame, lo copia internamente
y retorna el DataFrame enriquecido **sin modificar el original**.

Todas las funciones incluyen validaciones post-transformación (assertions)
para detectar errores silenciosos de forma temprana.

Funciones públicas:
    compute_review_features   — total_reviews, positive_ratio, review_label
    compute_time_features     — game_age_years, era, is_recent
    compute_price_features    — is_free, price_tier, log_price, price_per_hour
    compute_platform_features — n_platforms, is_multiplatform, platform_combo
    compute_vif               — Variance Inflation Factor para multicolinealidad
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Review features
# ---------------------------------------------------------------------------

def compute_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features derivadas de las reviews.

    Features creadas:
        - ``total_reviews``: suma de reviews positivas y negativas.
        - ``positive_ratio``: proporción de reviews positivas (NaN si 0 reviews).
        - ``review_label``: etiqueta categórica tipo Steam.
        - ``log_total_reviews``: ``log1p(total_reviews)`` para reducir skewness.

    Args:
        df: DataFrame con columnas ``positive`` y ``negative``.

    Returns:
        Copia del DataFrame con columnas adicionales.
    """
    result = df.copy()

    result["total_reviews"] = result["positive"] + result["negative"]
    result["positive_ratio"] = np.where(
        result["total_reviews"] > 0,
        result["positive"] / result["total_reviews"],
        np.nan,
    )
    result["log_total_reviews"] = np.log1p(result["total_reviews"])

    # Clasificación vectorizada
    conditions = [
        result["total_reviews"] < 10,
        result["positive_ratio"] >= 0.95,
        result["positive_ratio"] >= 0.85,
        result["positive_ratio"] >= 0.80,
        result["positive_ratio"] >= 0.70,
        result["positive_ratio"] >= 0.40,
        result["positive_ratio"] >= 0.20,
    ]
    choices = [
        "Sin Calificacion",
        "Overwhelmingly Positive",
        "Very Positive",
        "Mostly Positive",
        "Positive",
        "Mixed",
        "Mostly Negative",
    ]
    result["review_label"] = np.select(conditions, choices, default="Very Negative")

    # --- Validaciones ---
    assert len(result) == len(df), (
        f"La transformación cambió el número de filas: {len(df)} -> {len(result)}"
    )
    assert result["total_reviews"].ge(0).all(), (
        "total_reviews contiene valores negativos"
    )
    valid_ratio = result["positive_ratio"].dropna()
    assert valid_ratio.between(0.0, 1.0, inclusive="both").all(), (
        f"positive_ratio fuera de [0, 1]: "
        f"min={valid_ratio.min():.4f}, max={valid_ratio.max():.4f}"
    )

    return result


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------

def compute_time_features(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Calcula features temporales basadas en la fecha de lanzamiento.

    Features creadas:
        - ``game_age_years``: antigüedad del juego en años.
        - ``era``: era de lanzamiento categórica.
        - ``is_recent``: flag para juegos lanzados en los últimos 2 años.

    Args:
        df: DataFrame con columna ``release_date`` (datetime).
        reference_date: Fecha de referencia; por defecto ``pd.Timestamp.now()``.

    Returns:
        Copia del DataFrame con features temporales adicionales.
    """
    result = df.copy()
    if reference_date is None:
        reference_date = pd.Timestamp.now()

    if "release_date" in result.columns:
        result["game_age_years"] = (
            (reference_date - pd.to_datetime(result["release_date"], errors="coerce"))
            .dt.days
            / 365.25
        ).round(2)

    if "release_year" in result.columns:
        era_conditions = [
            result["release_year"].isna(),
            result["release_year"] < 2010,
            result["release_year"] < 2014,
            result["release_year"] < 2018,
            result["release_year"] < 2022,
        ]
        era_choices = [
            "Desconocido",
            "Pre-Greenlight (< 2010)",
            "Greenlight Era (2010-2013)",
            "Direct Era (2014-2017)",
            "Explosion (2018-2021)",
        ]
        result["era"] = np.select(era_conditions, era_choices, default="Actual (2022+)")

    if "game_age_years" in result.columns:
        result["is_recent"] = result["game_age_years"] <= 2.0

    # --- Validaciones ---
    assert len(result) == len(df), (
        f"La transformación cambió el número de filas: {len(df)} -> {len(result)}"
    )

    return result


# ---------------------------------------------------------------------------
# Price features
# ---------------------------------------------------------------------------

def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features relacionadas con el precio del juego.

    Features creadas:
        - ``is_free``: ``True`` si el precio es 0.
        - ``price_tier``: segmento de precio (Free → AAA).
        - ``log_price``: ``log1p(price)``.
        - ``price_per_hour``: precio / horas jugadas promedio.

    Args:
        df: DataFrame con columnas ``price`` y (opcionalmente)
            ``average_playtime_forever``.

    Returns:
        Copia del DataFrame con features de precio adicionales.
    """
    result = df.copy()

    result["is_free"] = result["price"] == 0.0
    result["log_price"] = np.log1p(result["price"])

    price_bins = [-0.01, 0.0, 5.0, 15.0, 30.0, 60.0, np.inf]
    price_labels = [
        "Free",
        "Low (<$5)",
        "Mid ($5-15)",
        "Standard ($15-30)",
        "Premium ($30-60)",
        "AAA (>$60)",
    ]
    result["price_tier"] = pd.cut(
        result["price"], bins=price_bins, labels=price_labels, right=True,
    )

    if "average_playtime_forever" in result.columns:
        hours_played = result["average_playtime_forever"] / 60.0
        result["price_per_hour"] = np.where(
            hours_played > 0,
            result["price"] / hours_played,
            np.nan,
        ).round(4)

    # --- Validaciones ---
    assert len(result) == len(df), (
        f"La transformación cambió el número de filas: {len(df)} -> {len(result)}"
    )
    assert result["price"].ge(0).all(), "price contiene valores negativos"
    assert result["is_free"].dtype == bool, "is_free no es de tipo bool"

    return result


# ---------------------------------------------------------------------------
# Platform features
# ---------------------------------------------------------------------------

def compute_platform_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features relacionadas con soporte de plataformas.

    Features creadas:
        - ``n_platforms``: número de plataformas soportadas (0-3).
        - ``is_multiplatform``: ``True`` si soporta más de una plataforma.
        - ``platform_combo``: combinación de plataformas como string.

    Args:
        df: DataFrame con columnas booleanas ``windows``, ``mac``, ``linux``.

    Returns:
        Copia del DataFrame con features de plataforma adicionales.
    """
    result = df.copy()
    plat_cols = [c for c in ["windows", "mac", "linux"] if c in result.columns]

    if plat_cols:
        plat_df = result[plat_cols].astype(int)
        result["n_platforms"] = plat_df.sum(axis=1)
        result["is_multiplatform"] = result["n_platforms"] > 1

        labels = {"windows": "Win", "mac": "Mac", "linux": "Lin"}
        combo_parts = pd.DataFrame(
            {
                col: result[col].astype(bool).map({True: labels[col], False: ""})
                for col in plat_cols
            }
        )
        result["platform_combo"] = combo_parts.apply(
            lambda row: "+".join(p for p in row if p), axis=1,
        )
        result["platform_combo"] = result["platform_combo"].replace("", "None")

    # --- Validaciones ---
    assert len(result) == len(df), (
        f"La transformación cambió el número de filas: {len(df)} -> {len(result)}"
    )
    if "n_platforms" in result.columns:
        assert result["n_platforms"].between(0, 3, inclusive="both").all(), (
            f"n_platforms fuera de [0, 3]"
        )

    return result


# ---------------------------------------------------------------------------
# VIF — Variance Inflation Factor
# ---------------------------------------------------------------------------

def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Calcula el Variance Inflation Factor para detectar multicolinealidad.

    Un VIF > 5 indica multicolinealidad moderada; VIF > 10 indica
    multicolinealidad severa que puede afectar la estabilidad de modelos
    lineales.

    Args:
        df: DataFrame con las columnas numéricas a evaluar.
        cols: Lista de nombres de columnas numéricas.

    Returns:
        DataFrame con columnas ``feature`` y ``VIF``, ordenado
        descendentemente por VIF.

    Raises:
        ImportError: Si ``statsmodels`` no está instalado.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        logger.warning(
            "statsmodels no está instalado. "
            "Ejecuta: pip install statsmodels"
        )
        existing = [c for c in cols if c in df.columns]
        return pd.DataFrame({"feature": existing, "VIF": [np.nan] * len(existing)})

    existing = [c for c in cols if c in df.columns]
    X = df[existing].dropna()

    # Agregar constante (intercepto) para el cálculo de VIF
    X = X.assign(_const=1.0)

    vif_data = []
    for i, col in enumerate(existing):
        vif_val = variance_inflation_factor(X.values, i)
        vif_data.append({"feature": col, "VIF": round(vif_val, 2)})

    return (
        pd.DataFrame(vif_data)
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )
