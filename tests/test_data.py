import numpy as np
import pandas as pd
import pytest
from src.data import prepare_features, validate_dataset


def test_validate_and_prepare_basic():
    """Verify standard happy-path validation and preparation with default threshold."""
    df = pd.DataFrame({
        "country": ["CountryA", "CountryB", "CountryC", "CountryD"],
        "rate_a": [1.0, 2.0, 3.0, 4.0],
        "rate_b": [4.0, 3.0, 2.0, 1.0],
    })
    country, numeric = validate_dataset(df)
    assert country == "country"
    assert set(numeric) == {"rate_a", "rate_b"}
    prepared = prepare_features(df, ["rate_a", "rate_b"])
    assert prepared["X"].shape == (4, 2)
    assert prepared["imputed_cells"] == 0
    assert prepared["retained_rows"] == 4
    assert prepared["excluded_rows"] == 0


def test_missing_and_blank_country_identifiers():
    """Verify rejection of null, empty, whitespace, and 'NaN' string country identifiers."""
    # Blank / whitespace
    df_blank = pd.DataFrame({
        "country": ["CountryA", "   ", "CountryC"],
        "rate_a": [1.0, 2.0, 3.0],
        "rate_b": [4.0, 5.0, 6.0],
    })
    with pytest.raises(ValueError, match="null or blank value"):
        validate_dataset(df_blank)

    # None / np.nan
    df_null = pd.DataFrame({
        "country": ["CountryA", None, "CountryC"],
        "rate_a": [1.0, 2.0, 3.0],
        "rate_b": [4.0, 5.0, 6.0],
    })
    with pytest.raises(ValueError, match="null or blank value"):
        validate_dataset(df_null)

    # Literal 'nan' string
    df_nan_str = pd.DataFrame({
        "country": ["CountryA", "NaN", "CountryC"],
        "rate_a": [1.0, 2.0, 3.0],
        "rate_b": [4.0, 5.0, 6.0],
    })
    with pytest.raises(ValueError, match="null or blank value"):
        validate_dataset(df_nan_str)


def test_duplicate_country_identifiers():
    """Verify detection of duplicate country names with explanatory guidance."""
    df_dup = pd.DataFrame({
        "country": ["France", "Germany", "France", "Italy"],
        "rate_a": [1.0, 2.0, 3.0, 4.0],
        "rate_b": [4.0, 3.0, 2.0, 1.0],
    })
    with pytest.raises(ValueError, match="Duplicate country identifiers detected: 'France'"):
        validate_dataset(df_dup)


def test_numeric_coercion_and_infinities():
    """Verify infinities and dirty strings are converted to NaN and handled appropriately."""
    df = pd.DataFrame({
        "country": ["A", "B", "C", "D"],
        "rate_a": [1.0, np.inf, 3.0, 4.0],
        "rate_b": ["4.0", "3.0", "bad_text", "1.0"],
        "rate_c": [10.0, 20.0, 30.0, 40.0],
        "rate_d": [100.0, 200.0, 300.0, 400.0],
    })
    # Country B has 1 NaN (from inf) out of 4 = 25% -> retained under 30% threshold
    # Country C has 1 NaN (from bad_text) out of 4 = 25% -> retained under 30% threshold
    prepared = prepare_features(df, ["rate_a", "rate_b", "rate_c", "rate_d"], max_missing_fraction=0.30)
    assert prepared["retained_rows"] == 4
    assert prepared["imputed_cells"] == 2
    assert np.isfinite(prepared["X"]).all()


def test_missingness_threshold_boundary_semantics():
    """Verify behavior for rows below, exactly at, and above the missingness threshold."""
    # 4 features:
    # Row A: 0/4 missing (0%) -> below threshold
    # Row B: 1/4 missing (25%) -> below threshold
    # Row C: 2/4 missing (50%) -> exactly at threshold (when threshold = 0.50)
    # Row D: 3/4 missing (75%) -> above threshold
    # Row E: 0/4 missing (0%) -> below threshold
    df = pd.DataFrame({
        "country": ["CountryA", "CountryB", "CountryC", "CountryD", "CountryE"],
        "f1": [1.0, None, None, None, 5.0],
        "f2": [2.0, 2.0, None, None, 4.0],
        "f3": [1.0, 3.0, 5.0, None, 7.0],
        "f4": [4.0, 4.0, 4.0, 4.0, 2.0],
    })

    # Test at threshold = 0.50 (50%):
    # Retained: A (0%), B (25%), C (50% - exactly at threshold), E (0%)
    # Excluded: D (75%)
    prep_50 = prepare_features(df, ["f1", "f2", "f3", "f4"], max_missing_fraction=0.50)
    assert prep_50["retained_rows"] == 4
    assert prep_50["excluded_rows"] == 1
    assert "CountryC" in [df.loc[i, "country"] for i in prep_50["row_indices"]]
    assert "CountryD" not in [df.loc[i, "country"] for i in prep_50["row_indices"]]

    # Audit check on Row D
    audit = prep_50["exclusions_df"]
    assert len(audit) == 1
    assert audit.iloc[0]["country"] == "CountryD"
    assert audit.iloc[0]["missing_count"] == 3
    assert audit.iloc[0]["missing_fraction"] == 0.75
    assert audit.iloc[0]["reason"] == "EXCEEDS_MISSING_THRESHOLD"

    # Test at default threshold = 0.30 (30%):
    # Retained: A (0%), B (25%), E (0%)
    # Excluded: C (50%), D (75%)
    prep_30 = prepare_features(df, ["f1", "f2", "f3", "f4"], max_missing_fraction=0.30)
    assert prep_30["retained_rows"] == 3
    assert prep_30["excluded_rows"] == 2
    assert set(prep_30["exclusions_df"]["country"]) == {"CountryC", "CountryD"}


def test_median_imputation_values():
    """Verify median imputation correctly applies the column median of retained rows."""
    # Retained rows: A, B, C, D (all 0 or 25% missing <= 30%)
    # Column f1 has [10.0, 20.0, None, 40.0] -> observed: 10, 20, 40 -> median = 20.0
    df = pd.DataFrame({
        "country": ["A", "B", "C", "D"],
        "f1": [10.0, 20.0, None, 40.0],
        "f2": [1.0, 2.0, 3.0, 4.0],
        "f3": [5.0, 6.0, 7.0, 8.0],
        "f4": [15.0, 16.0, 17.0, 18.0],
    })
    prepared = prepare_features(df, ["f1", "f2", "f3", "f4"], max_missing_fraction=0.30)
    assert prepared["imputed_cells"] == 1
    # Check imputed value via the fitted imputer
    imputed_f1 = prepared["imputer"].statistics_[0]
    assert imputed_f1 == 20.0


def test_fewer_than_three_usable_rows_error():
    """Verify clear ValueError when filtering drops rows below the minimum requirement."""
    df = pd.DataFrame({
        "country": ["A", "B", "C", "D"],
        "f1": [1.0, None, None, None],
        "f2": [2.0, 2.0, None, None],
        "f3": [3.0, None, None, None],
    })
    # Threshold 0.20: Only Country A has 0% missing, others have >= 33% missing -> only 1 row remains
    with pytest.raises(ValueError, match="Fewer than three usable country rows remain"):
        prepare_features(df, ["f1", "f2", "f3"], max_missing_fraction=0.20)


def test_constant_feature_rejection():
    """Verify constant feature columns are rejected with an informative error."""
    df = pd.DataFrame({
        "country": ["A", "B", "C", "D"],
        "f1": [5.0, 5.0, 5.0, 5.0],
        "f2": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError, match="Remove constant feature columns: \\['f1'\\]"):
        prepare_features(df, ["f1", "f2"])


def test_stable_row_order_and_correspondence():
    """Verify retained rows maintain exact source index and ordering."""
    df = pd.DataFrame({
        "country": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "f1": [10.0, None, 30.0, None, 50.0],
        "f2": [100.0, None, 300.0, 400.0, 500.0],
        "f3": [1.0, None, 3.0, 4.0, 5.0],
    }, index=[10, 20, 30, 40, 50])

    # Beta has 100% missing -> excluded
    # Delta has 33.3% missing -> excluded at 0.30 threshold
    # Alpha (10), Gamma (30), Epsilon (50) retained
    prepared = prepare_features(df, ["f1", "f2", "f3"], max_missing_fraction=0.30)
    assert prepared["row_indices"] == [10, 30, 50]
    retained_df = df.loc[prepared["row_indices"]]
    assert list(retained_df["country"]) == ["Alpha", "Gamma", "Epsilon"]


def test_invalid_missingness_threshold_parameter():
    """Verify threshold outside [0, 1] is rejected."""
    df = pd.DataFrame({
        "country": ["A", "B", "C"],
        "f1": [1.0, 2.0, 3.0],
        "f2": [4.0, 5.0, 6.0],
    })
    with pytest.raises(ValueError, match="max_missing_fraction must be between 0.0 and 1.0"):
        prepare_features(df, ["f1", "f2"], max_missing_fraction=1.5)
