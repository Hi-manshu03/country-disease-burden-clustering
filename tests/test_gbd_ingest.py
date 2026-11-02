"""Unit tests for GBD long-to-wide data ingestion pipeline (P28-008A).

Tests verify schema validation, dimension filtering, duplicate detection,
long-to-wide pivoting, and downstream compatibility with existing clustering modules.
All tests use synthetic long-format GBD fixtures without requiring external downloads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.gbd_transform import transform_gbd_long_to_wide, DEFAULT_GBD_CAUSES
from src.data import validate_dataset, prepare_features
from src.clustering import evaluate_kmeans, evaluate_kmedians


@pytest.fixture
def synthetic_gbd_long() -> pd.DataFrame:
    """Fixture returning a standard long-format GBD DataFrame."""
    records = []
    countries = ["Japan", "Nigeria", "Brazil", "Germany", "Global", "Western Europe"]
    causes = ["Cardiovascular diseases", "Neoplasms", "Enteric infections"]
    
    val_map = {
        ("Japan", "Cardiovascular diseases"): 150.2,
        ("Japan", "Neoplasms"): 220.5,
        ("Japan", "Enteric infections"): 12.1,
        ("Nigeria", "Cardiovascular diseases"): 280.4,
        ("Nigeria", "Neoplasms"): 110.0,
        ("Nigeria", "Enteric infections"): 340.8,
        ("Brazil", "Cardiovascular diseases"): 230.1,
        ("Brazil", "Neoplasms"): 180.4,
        ("Brazil", "Enteric infections"): 65.3,
        ("Germany", "Cardiovascular diseases"): 190.5,
        ("Germany", "Neoplasms"): 240.2,
        ("Germany", "Enteric infections"): 15.0,
        ("Global", "Cardiovascular diseases"): 250.0,
        ("Global", "Neoplasms"): 200.0,
        ("Global", "Enteric infections"): 100.0,
        ("Western Europe", "Cardiovascular diseases"): 180.0,
        ("Western Europe", "Neoplasms"): 230.0,
        ("Western Europe", "Enteric infections"): 14.0,
    }
    
    for (loc, cause), v in val_map.items():
        records.append({
            "measure_name": "DALYs (Disability-Adjusted Life Years)",
            "location_name": loc,
            "sex_name": "Both",
            "age_name": "Age-standardized",
            "cause_name": cause,
            "metric_name": "Rate",
            "year": 2021,
            "val": v,
            "upper": v * 1.1,
            "lower": v * 0.9,
        })
    return pd.DataFrame(records)


def test_transform_gbd_long_to_wide_basic(synthetic_gbd_long):
    """Verify standard long-to-wide pivot produces correct country rows and feature columns."""
    wide = transform_gbd_long_to_wide(synthetic_gbd_long)
    
    # Global and Western Europe must be filtered out
    expected_countries = {"Japan", "Nigeria", "Brazil", "Germany"}
    assert set(wide["country"]) == expected_countries
    assert len(wide) == 4
    
    # Feature columns check
    expected_features = {"Cardiovascular diseases", "Neoplasms", "Enteric infections"}
    assert expected_features.issubset(set(wide.columns))
    
    # Point estimate verification for Japan
    japan_row = wide[wide["country"] == "Japan"].iloc[0]
    assert japan_row["Cardiovascular diseases"] == 150.2
    assert japan_row["Neoplasms"] == 220.5
    assert japan_row["Enteric infections"] == 12.1


def test_transform_missing_required_columns():
    """Verify ValueError is raised if essential GBD columns are missing."""
    df_bad = pd.DataFrame({"some_col": [1, 2], "val": [10.0, 20.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        transform_gbd_long_to_wide(df_bad)


def test_transform_empty_dataframe():
    """Verify ValueError is raised on empty input."""
    with pytest.raises(ValueError, match="is empty"):
        transform_gbd_long_to_wide(pd.DataFrame())


def test_transform_filters_global_and_regional_aggregates(synthetic_gbd_long):
    """Verify regional super-aggregates are excluded when filter_aggregates=True."""
    wide = transform_gbd_long_to_wide(synthetic_gbd_long, filter_aggregates=True)
    assert "Global" not in wide["country"].values
    assert "Western Europe" not in wide["country"].values


def test_transform_detects_duplicate_country_cause_observations(synthetic_gbd_long):
    """Verify duplicate (location, cause) observations trigger a clear ValueError."""
    # Append a duplicate record for Japan - Cardiovascular diseases
    dup_row = pd.DataFrame([{
        "measure_name": "DALYs (Disability-Adjusted Life Years)",
        "location_name": "Japan",
        "sex_name": "Both",
        "age_name": "Age-standardized",
        "cause_name": "Cardiovascular diseases",
        "metric_name": "Rate",
        "year": 2021,
        "val": 999.9,
    }])
    df_with_dup = pd.concat([synthetic_gbd_long, dup_row], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate \\(location, cause\\) observations detected"):
        transform_gbd_long_to_wide(df_with_dup)


def test_transform_filter_consistency_year(synthetic_gbd_long):
    """Verify transformation filters multi-year datasets properly."""
    # Add 2019 records to synthetic data
    older_records = synthetic_gbd_long.copy()
    older_records["year"] = 2019
    older_records["val"] = older_records["val"] * 0.95
    multi_year = pd.concat([synthetic_gbd_long, older_records], ignore_index=True)
    
    # When target_year=2019 specified
    wide_2019 = transform_gbd_long_to_wide(multi_year, target_year=2019)
    japan_2019 = wide_2019[wide_2019["country"] == "Japan"].iloc[0]["Cardiovascular diseases"]
    assert np.isclose(japan_2019, 150.2 * 0.95)
    
    # When target_year=None, defaults to latest year (2021)
    wide_latest = transform_gbd_long_to_wide(multi_year)
    japan_latest = wide_latest[wide_latest["country"] == "Japan"].iloc[0]["Cardiovascular diseases"]
    assert np.isclose(japan_latest, 150.2)


def test_transform_rejects_mismatched_measurement_dimension(synthetic_gbd_long):
    """Do not keep source rows when no record matches the required GBD dimension."""
    incorrect_metric = synthetic_gbd_long.copy()
    incorrect_metric["metric_name"] = "Number"
    with pytest.raises(ValueError, match="No records match the required GBD metric_name"):
        transform_gbd_long_to_wide(incorrect_metric)


def test_transform_rejects_requested_year_not_in_source(synthetic_gbd_long):
    with pytest.raises(ValueError, match="No records remain after applying standard GBD dimension filters"):
        transform_gbd_long_to_wide(synthetic_gbd_long, target_year=2019)


def test_transform_compatibility_with_validate_and_prepare(synthetic_gbd_long):
    """Verify transformed wide DataFrame satisfies validate_dataset and prepare_features."""
    wide = transform_gbd_long_to_wide(synthetic_gbd_long)
    
    # 1. Validation contract
    country_col, numeric_cols = validate_dataset(wide)
    assert country_col == "country"
    assert len(numeric_cols) == 3
    
    # 2. Preprocessing contract
    prepared = prepare_features(wide, numeric_cols)
    assert prepared["retained_rows"] == 4
    assert prepared["excluded_rows"] == 0
    assert prepared["X"].shape == (4, 3)
    assert np.isfinite(prepared["X"]).all()


def test_transform_compatibility_with_kmeans_and_kmedians(synthetic_gbd_long):
    """Verify transformed dataset can run end-to-end through clustering algorithms."""
    wide = transform_gbd_long_to_wide(synthetic_gbd_long)
    country_col, numeric_cols = validate_dataset(wide)
    prepared = prepare_features(wide, numeric_cols)
    
    # Run K-Means
    km_res = evaluate_kmeans(prepared["X"], [2, 3])
    assert "metrics" in km_res
    assert len(km_res["metrics"]) == 2
    
    # Run K-Medians
    kmed_res = evaluate_kmedians(prepared["X"], [2, 3])
    assert "metrics" in kmed_res
    assert len(kmed_res["metrics"]) == 2
