"""Tests for P28-007A: Application stabilization, input validation, and deployment readiness."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

from src.data import validate_dataset, prepare_features
from src.clustering import evaluate_kmeans, evaluate_kmedians
from src.visualization import make_evaluation_curves


APP_PATH = str(Path(__file__).parent.parent / "app.py")


# ── Phase C: Input validation & error handling unit tests ────────────────────

def test_validate_dataset_rejects_missing_country_identifier():
    """Verify that a CSV lacking a country identifier is rejected with an actionable error."""
    df_no_country = pd.DataFrame({
        "meas_a": [1.0, 2.0, 3.0, 4.0],
        "meas_b": [10.0, 20.0, 30.0, 40.0],
        "meas_c": [100.0, 200.0, 300.0, 400.0],
    })
    with pytest.raises(ValueError, match="No country identifier column found"):
        validate_dataset(df_no_country)


def test_validate_dataset_rejects_single_numeric_feature():
    """Verify that a CSV with fewer than two usable numeric features is rejected."""
    df_one_feature = pd.DataFrame({
        "country": ["CountryA", "CountryB", "CountryC", "CountryD"],
        "rate_a": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ValueError, match="At least two usable numeric feature columns are required"):
        validate_dataset(df_one_feature)


def test_validate_dataset_accepts_alternative_candidate_names():
    """Verify supported synonyms like 'location', 'entity', 'iso3' are accepted."""
    for col_name in ["location", "entity", "iso3", "country_name"]:
        df = pd.DataFrame({
            col_name: ["CountryA", "CountryB", "CountryC"],
            "rate_a": [1.0, 2.0, 3.0],
            "rate_b": [4.0, 5.0, 6.0],
        })
        country_col, numeric = validate_dataset(df)
        assert country_col == col_name
        assert set(numeric) == {"rate_a", "rate_b"}


# ── Phase D: Methodological consistency unit tests ───────────────────────────

def test_kmedians_evaluation_curve_labels_manhattan_not_wcss():
    """Verify that K-Medians elbow curve is labeled as Manhattan/L1 objective and does NOT say WCSS."""
    metrics_df = pd.DataFrame({
        "k": [2, 3, 4],
        "kmedians_l1_objective": [120.0, 85.0, 60.0],
        "silhouette_score": [0.55, 0.48, 0.35],
    })
    fig = make_evaluation_curves(metrics_df, algorithm="K-Medians (Manhattan)")
    
    # Check subplot titles in annotations or layout
    layout_text = " ".join([a.text for a in fig.layout.annotations if hasattr(a, "text")])
    assert "L1 Objective (Manhattan)" in layout_text
    assert "WCSS" not in layout_text, f"K-Medians panel title must not contain 'WCSS': {layout_text}"


def test_evaluate_kmeans_safe_on_degenerate_points():
    """Verify evaluate_kmeans does not crash on degenerate duplicate data points."""
    # 6 points but all identical
    X = np.ones((6, 3))
    res = evaluate_kmeans(X, [2, 3])
    assert "metrics" in res
    assert len(res["metrics"]) == 2
    # Silhouette score on identical points is safe (nan)
    assert np.isnan(res["metrics"]["silhouette_score"].iloc[0]) or isinstance(
        res["metrics"]["silhouette_score"].iloc[0], float
    )


def test_evaluate_kmedians_safe_on_degenerate_points():
    """Verify evaluate_kmedians does not crash on degenerate duplicate data points."""
    X = np.ones((6, 3))
    res = evaluate_kmedians(X, [2, 3])
    assert "metrics" in res
    assert len(res["metrics"]) == 2


def test_evaluate_kmeans_raises_when_all_k_invalid():
    """Verify evaluate_kmeans raises ValueError when all k >= n_samples."""
    X = np.ones((4, 3))
    with pytest.raises(ValueError, match="Choose k values with 2 <= k < number of usable rows"):
        evaluate_kmeans(X, [4, 5, 6])


def test_evaluate_kmedians_raises_when_all_k_invalid():
    """Verify evaluate_kmedians raises ValueError when all k >= n_samples."""
    X = np.ones((4, 3))
    with pytest.raises(ValueError, match="Choose k values with 2 <= k < number of usable rows"):
        evaluate_kmedians(X, [4, 5, 6])


# ── Phase B & C: Streamlit integration tests ──────────────────────────────────

def test_app_invalid_country_csv_displays_error():
    """Verify app shows an error without crashing when CSV has no country column."""
    csv_bytes = b"""feat_a,feat_b,feat_c
1.0,2.0,3.0
4.0,5.0,6.0
7.0,8.0,9.0
10.0,11.0,12.0
"""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.file_uploader[0].upload("no_country.csv", csv_bytes).run()
    
    # Should display st.error indicating missing country identifier
    assert len(at.error) >= 1
    assert "No country identifier column found" in at.error[0].value
    # Analysis controls should not be rendered
    assert len(at.button) == 0


def test_app_omits_invalid_k_with_warning():
    """Verify app warns when some requested k values exceed sample size."""
    # 4 countries -> valid k are strictly < 4, so k=2, 3 are valid, k=4 is omitted
    csv_bytes = b"""country,meas_a,meas_b,meas_c
Country1,10,20,30
Country2,40,50,60
Country3,70,80,90
Country4,15,25,35
"""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.file_uploader[0].upload("four_countries.csv", csv_bytes).run()
    
    # Default k is [2, 3, 4]
    at.button[0].click().run()
    
    # Verify warning about omitted k
    warning_texts = [w.value for w in at.warning]
    assert any("were omitted because k must be strictly less" in t for t in warning_texts)
    
    # Check that available_k only has 2 and 3
    cached = at.session_state["clustering_result"]
    assert list(cached["labels_by_k"].keys()) == [2, 3]

