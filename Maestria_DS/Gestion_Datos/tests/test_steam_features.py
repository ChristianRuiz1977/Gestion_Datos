"""Tests unitarios para src/features/steam_features.py.

Cubre las funciones de feature engineering migradas del notebook 02,
verificando preservación de filas, rangos válidos y tipos de datos.

Ejecutar con:
    pytest tests/test_steam_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.features.steam_features import (
    compute_platform_features,
    compute_price_features,
    compute_review_features,
    compute_time_features,
    compute_vif,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """DataFrame de muestra representativo para todas las pruebas."""
    return pd.DataFrame(
        {
            "AppID": ["1", "2", "3", "4", "5"],
            "name": ["Game A", "Game B", "Game C", "Game D", "Game E"],
            "price": [0.0, 9.99, 29.99, 59.99, 4.99],
            "positive": [100, 5000, 0, 50, 10],
            "negative": [10, 500, 0, 50, 2],
            "windows": [True, True, True, True, False],
            "mac": [False, True, False, True, False],
            "linux": [False, False, True, True, False],
            "release_date": pd.to_datetime(
                ["2020-01-15", "2015-06-01", "2023-11-20", "2010-03-10", "2024-12-01"]
            ),
            "release_year": [2020, 2015, 2023, 2010, 2024],
            "average_playtime_forever": [300, 1200, 0, 60, 10],
        }
    )


@pytest.fixture
def large_sample_df() -> pd.DataFrame:
    """DataFrame más grande para tests de VIF."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "price": np.random.uniform(0, 60, n),
            "positive": np.random.randint(0, 10000, n),
            "negative": np.random.randint(0, 2000, n),
            "achievements": np.random.randint(0, 100, n),
            "n_languages": np.random.randint(1, 30, n),
        }
    )


# ===========================================================================
# compute_review_features
# ===========================================================================


class TestComputeReviewFeatures:
    """Tests para compute_review_features."""

    def test_preserves_row_count(self, sample_df):
        result = compute_review_features(sample_df)
        assert len(result) == len(sample_df)

    def test_does_not_mutate_input(self, sample_df):
        original_cols = list(sample_df.columns)
        compute_review_features(sample_df)
        assert list(sample_df.columns) == original_cols

    def test_total_reviews_nonnegative(self, sample_df):
        result = compute_review_features(sample_df)
        assert result["total_reviews"].ge(0).all()

    def test_positive_ratio_bounds(self, sample_df):
        result = compute_review_features(sample_df)
        valid = result["positive_ratio"].dropna()
        assert valid.between(0.0, 1.0).all()

    def test_positive_ratio_nan_when_zero_reviews(self, sample_df):
        result = compute_review_features(sample_df)
        zero_mask = result["total_reviews"] == 0
        assert result.loc[zero_mask, "positive_ratio"].isna().all()

    def test_review_label_is_string(self, sample_df):
        result = compute_review_features(sample_df)
        assert pd.api.types.is_string_dtype(result["review_label"])

    def test_log_total_reviews_nonnegative(self, sample_df):
        result = compute_review_features(sample_df)
        assert result["log_total_reviews"].ge(0).all()

    def test_known_ratio(self):
        """100 positive, 0 negative -> ratio = 1.0."""
        df = pd.DataFrame({"positive": [100], "negative": [0]})
        result = compute_review_features(df)
        assert result["positive_ratio"].iloc[0] == 1.0

    def test_even_split_ratio(self):
        """50 positive, 50 negative -> ratio = 0.5."""
        df = pd.DataFrame({"positive": [50], "negative": [50]})
        result = compute_review_features(df)
        assert result["positive_ratio"].iloc[0] == pytest.approx(0.5)

    def test_review_label_categories(self, sample_df):
        result = compute_review_features(sample_df)
        valid_labels = {
            "Sin Calificacion",
            "Overwhelmingly Positive",
            "Very Positive",
            "Mostly Positive",
            "Positive",
            "Mixed",
            "Mostly Negative",
            "Very Negative",
        }
        assert set(result["review_label"].unique()).issubset(valid_labels)


# ===========================================================================
# compute_time_features
# ===========================================================================


class TestComputeTimeFeatures:
    """Tests para compute_time_features."""

    def test_preserves_row_count(self, sample_df):
        result = compute_time_features(sample_df)
        assert len(result) == len(sample_df)

    def test_does_not_mutate_input(self, sample_df):
        original_cols = list(sample_df.columns)
        compute_time_features(sample_df)
        assert list(sample_df.columns) == original_cols

    def test_game_age_positive_for_past_dates(self, sample_df):
        ref = pd.Timestamp("2026-02-13")
        result = compute_time_features(sample_df, reference_date=ref)
        assert result["game_age_years"].ge(0).all()

    def test_era_column_created(self, sample_df):
        result = compute_time_features(sample_df)
        assert "era" in result.columns

    def test_is_recent_is_bool(self, sample_df):
        result = compute_time_features(sample_df)
        assert result["is_recent"].dtype == bool

    def test_era_categories(self, sample_df):
        result = compute_time_features(sample_df)
        valid_eras = {
            "Desconocido",
            "Pre-Greenlight (< 2010)",
            "Greenlight Era (2010-2013)",
            "Direct Era (2014-2017)",
            "Explosion (2018-2021)",
            "Actual (2022+)",
        }
        assert set(result["era"].unique()).issubset(valid_eras)

    def test_custom_reference_date(self, sample_df):
        """Con referencia en 2025, un juego de 2024 tiene < 2 años."""
        ref = pd.Timestamp("2025-06-01")
        result = compute_time_features(sample_df, reference_date=ref)
        mask_2024 = sample_df["release_year"] == 2024
        assert result.loc[mask_2024, "is_recent"].all()


# ===========================================================================
# compute_price_features
# ===========================================================================


class TestComputePriceFeatures:
    """Tests para compute_price_features."""

    def test_preserves_row_count(self, sample_df):
        result = compute_price_features(sample_df)
        assert len(result) == len(sample_df)

    def test_does_not_mutate_input(self, sample_df):
        original_cols = list(sample_df.columns)
        compute_price_features(sample_df)
        assert list(sample_df.columns) == original_cols

    def test_is_free_for_zero_price(self, sample_df):
        result = compute_price_features(sample_df)
        zero_mask = sample_df["price"] == 0.0
        assert result.loc[zero_mask, "is_free"].all()

    def test_is_free_false_for_nonzero(self, sample_df):
        result = compute_price_features(sample_df)
        paid_mask = sample_df["price"] > 0
        assert not result.loc[paid_mask, "is_free"].any()

    def test_is_free_is_bool(self, sample_df):
        result = compute_price_features(sample_df)
        assert result["is_free"].dtype == bool

    def test_log_price_nonnegative(self, sample_df):
        result = compute_price_features(sample_df)
        assert result["log_price"].ge(0).all()

    def test_price_tier_no_unexpected_nans(self, sample_df):
        result = compute_price_features(sample_df)
        assert result["price_tier"].notna().all()

    def test_price_per_hour_nan_when_zero_playtime(self, sample_df):
        result = compute_price_features(sample_df)
        zero_playtime = sample_df["average_playtime_forever"] == 0
        assert result.loc[zero_playtime, "price_per_hour"].isna().all()

    def test_price_tier_categories(self, sample_df):
        result = compute_price_features(sample_df)
        valid_tiers = {
            "Free", "Low (<$5)", "Mid ($5-15)",
            "Standard ($15-30)", "Premium ($30-60)", "AAA (>$60)",
        }
        actual = set(result["price_tier"].dropna().astype(str).unique())
        assert actual.issubset(valid_tiers)


# ===========================================================================
# compute_platform_features
# ===========================================================================


class TestComputePlatformFeatures:
    """Tests para compute_platform_features."""

    def test_preserves_row_count(self, sample_df):
        result = compute_platform_features(sample_df)
        assert len(result) == len(sample_df)

    def test_does_not_mutate_input(self, sample_df):
        original_cols = list(sample_df.columns)
        compute_platform_features(sample_df)
        assert list(sample_df.columns) == original_cols

    def test_n_platforms_range(self, sample_df):
        result = compute_platform_features(sample_df)
        assert result["n_platforms"].between(0, 3).all()

    def test_is_multiplatform_is_bool(self, sample_df):
        result = compute_platform_features(sample_df)
        assert result["is_multiplatform"].dtype == bool

    def test_multiplatform_consistency(self, sample_df):
        result = compute_platform_features(sample_df)
        multi_mask = result["n_platforms"] > 1
        assert (result.loc[multi_mask, "is_multiplatform"]).all()
        assert not (result.loc[~multi_mask, "is_multiplatform"]).any()

    def test_platform_combo_not_empty(self, sample_df):
        result = compute_platform_features(sample_df)
        assert (result["platform_combo"] != "").all()

    def test_windows_only(self):
        df = pd.DataFrame(
            {"windows": [True], "mac": [False], "linux": [False]}
        )
        result = compute_platform_features(df)
        assert result["platform_combo"].iloc[0] == "Win"
        assert result["n_platforms"].iloc[0] == 1

    def test_all_platforms(self):
        df = pd.DataFrame(
            {"windows": [True], "mac": [True], "linux": [True]}
        )
        result = compute_platform_features(df)
        assert result["platform_combo"].iloc[0] == "Win+Mac+Lin"
        assert result["n_platforms"].iloc[0] == 3

    def test_no_platforms(self):
        df = pd.DataFrame(
            {"windows": [False], "mac": [False], "linux": [False]}
        )
        result = compute_platform_features(df)
        assert result["platform_combo"].iloc[0] == "None"
        assert result["n_platforms"].iloc[0] == 0


# ===========================================================================
# compute_vif
# ===========================================================================


class TestComputeVif:
    """Tests para compute_vif."""

    def test_returns_dataframe(self, large_sample_df):
        cols = ["price", "positive", "negative"]
        result = compute_vif(large_sample_df, cols)
        assert isinstance(result, pd.DataFrame)

    def test_has_feature_and_vif_columns(self, large_sample_df):
        cols = ["price", "positive"]
        result = compute_vif(large_sample_df, cols)
        assert "feature" in result.columns
        assert "VIF" in result.columns

    def test_all_features_present(self, large_sample_df):
        cols = ["price", "positive", "negative"]
        result = compute_vif(large_sample_df, cols)
        assert set(result["feature"]) == set(cols)

    def test_vif_positive(self, large_sample_df):
        cols = ["price", "positive", "negative"]
        result = compute_vif(large_sample_df, cols)
        assert result["VIF"].gt(0).all()

    def test_skips_missing_columns(self, large_sample_df):
        cols = ["price", "nonexistent_col"]
        result = compute_vif(large_sample_df, cols)
        assert "nonexistent_col" not in result["feature"].values

    def test_high_vif_for_collinear_features(self):
        """Dos columnas perfectamente correlacionadas deben tener VIF alto."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        df = pd.DataFrame(
            {
                "x": x,
                "x_copy": x + np.random.normal(0, 0.01, n),
                "independent": np.random.randn(n),
            }
        )
        result = compute_vif(df, ["x", "x_copy", "independent"])
        x_vif = result.loc[result["feature"] == "x", "VIF"].iloc[0]
        assert x_vif > 10, f"VIF de columnas colineales deberia ser > 10, got {x_vif}"


# ===========================================================================
# Pipeline integration
# ===========================================================================


class TestPipelineIntegration:
    """Test de integración: encadenar todas las funciones."""

    def test_full_pipeline(self, sample_df):
        result = (
            sample_df
            .pipe(compute_review_features)
            .pipe(compute_time_features)
            .pipe(compute_price_features)
            .pipe(compute_platform_features)
        )
        assert len(result) == len(sample_df)
        # Verificar que todas las features esperadas existen
        expected_cols = [
            "total_reviews", "positive_ratio", "review_label",
            "game_age_years", "era", "is_recent",
            "is_free", "price_tier", "log_price",
            "n_platforms", "is_multiplatform", "platform_combo",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Falta columna: {col}"
