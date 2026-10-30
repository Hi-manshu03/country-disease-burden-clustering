"""Tests for P28-005: clustering methodology, evaluation curves, K-Medians,
and distributional cluster profiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clustering import (
    KMedians,
    compute_cluster_profiles,
    evaluate_kmedians,
    evaluate_kmeans,
)
from src.visualization import make_evaluation_curves, make_pca_loadings_figure


# ── Helpers ──────────────────────────────────────────────────────────────────

def _two_cluster_data(seed: int = 0) -> np.ndarray:
    """Return a clean 40×3 dataset with two well-separated Gaussian clusters."""
    rng = np.random.default_rng(seed)
    return np.vstack([
        rng.normal(loc=-4, scale=0.4, size=(20, 3)),
        rng.normal(loc=4, scale=0.4, size=(20, 3)),
    ])


# ── KMedians unit tests ───────────────────────────────────────────────────────

class TestKMedians:
    """Unit tests for the custom KMedians implementation."""

    def test_basic_two_cluster_separation(self):
        """K-Medians must correctly separate two well-defined Gaussian clusters."""
        X = _two_cluster_data(seed=1)
        model = KMedians(n_clusters=2, random_state=0)
        labels = model.fit_predict(X)
        assert labels.shape == (40,), "Labels must be (n_samples,)."
        # The two Gaussian blobs should map cleanly to two distinct label groups.
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 2, f"Expected 2 clusters, got {len(unique_labels)}."
        # First 20 observations should share a single label.
        assert len(np.unique(labels[:20])) == 1, "First 20 points should be in one cluster."
        assert len(np.unique(labels[20:])) == 1, "Last 20 points should be in one cluster."
        assert labels[0] != labels[20], "The two blobs must be in different clusters."

    def test_fit_predict_consistency(self):
        """fit() and fit_predict() must produce identical labels."""
        X = _two_cluster_data(seed=2)
        m1 = KMedians(n_clusters=2, random_state=7)
        m2 = KMedians(n_clusters=2, random_state=7)
        labels_fp = m1.fit_predict(X)
        m2.fit(X)
        np.testing.assert_array_equal(labels_fp, m2.labels_)

    def test_cluster_medians_are_actual_medians(self):
        """Stored cluster_medians_ must equal the feature-wise median of assigned points."""
        X = _two_cluster_data(seed=3)
        model = KMedians(n_clusters=2, random_state=0)
        model.fit(X)
        for c in range(2):
            mask = model.labels_ == c
            expected_median = np.median(X[mask], axis=0)
            np.testing.assert_allclose(
                model.cluster_medians_[c], expected_median, atol=1e-10,
                err_msg=f"Cluster {c} median mismatch.",
            )

    def test_l1_objective_is_finite_and_positive(self):
        """L1 objective must be finite, non-negative, and non-zero for well-separated data."""
        X = _two_cluster_data(seed=4)
        model = KMedians(n_clusters=2, random_state=0)
        model.fit(X)
        assert np.isfinite(model.l1_objective_), "L1 objective must be finite."
        assert model.l1_objective_ > 0, "L1 objective must be positive."

    def test_n_iter_upper_bound(self):
        """n_iter_ must be between 1 and max_iter (inclusive)."""
        X = _two_cluster_data(seed=5)
        model = KMedians(n_clusters=2, max_iter=50, random_state=0)
        model.fit(X)
        assert 1 <= model.n_iter_ <= 50, f"n_iter_ {model.n_iter_} out of [1, 50]."

    def test_raises_on_n_clusters_less_than_two(self):
        """n_clusters < 2 must raise ValueError."""
        X = _two_cluster_data(seed=6)
        with pytest.raises(ValueError, match="n_clusters must be >= 2"):
            KMedians(n_clusters=1).fit_predict(X)

    def test_raises_when_n_clusters_equals_n_samples(self):
        """n_clusters >= n_samples must raise ValueError."""
        X = np.ones((5, 2))
        with pytest.raises(ValueError, match="strictly less than number of samples"):
            KMedians(n_clusters=5).fit_predict(X)

    def test_determinism_with_fixed_random_state(self):
        """Two runs with the same random_state must produce identical labels."""
        X = _two_cluster_data(seed=8)
        labels_a = KMedians(n_clusters=3, random_state=42).fit_predict(X)
        labels_b = KMedians(n_clusters=3, random_state=42).fit_predict(X)
        np.testing.assert_array_equal(labels_a, labels_b)

    def test_no_empty_clusters_on_clean_data(self):
        """All requested clusters must be populated for well-separated clean data."""
        X = _two_cluster_data(seed=9)
        model = KMedians(n_clusters=2, random_state=0)
        model.fit(X)
        for c in range(2):
            assert np.any(model.labels_ == c), f"Cluster {c} is empty."

    def test_manhattan_objective_matches_manual_calculation(self):
        """Verify the L1 objective stored equals the manual per-point Manhattan sum."""
        X = _two_cluster_data(seed=10)
        model = KMedians(n_clusters=2, random_state=0)
        model.fit(X)
        # Manual calculation
        assigned_medians = model.cluster_medians_[model.labels_]
        manual_objective = float(np.sum(np.abs(X - assigned_medians)))
        assert abs(model.l1_objective_ - manual_objective) < 1e-6


# ── evaluate_kmedians tests ───────────────────────────────────────────────────

class TestEvaluateKMedians:
    """Integration tests for the evaluate_kmedians function."""

    def test_returns_expected_structure(self):
        """evaluate_kmedians must return algorithm, metrics, labels_by_k, models keys."""
        X = _two_cluster_data(seed=11)
        result = evaluate_kmedians(X, [2, 3, 4])
        assert "algorithm" in result
        assert "metrics" in result
        assert "labels_by_k" in result
        assert "models" in result

    def test_algorithm_label_is_manhattan(self):
        """The algorithm label must reference Manhattan distance."""
        X = _two_cluster_data(seed=12)
        result = evaluate_kmedians(X, [2, 3])
        assert "Manhattan" in result["algorithm"], (
            f"Expected 'Manhattan' in algorithm label, got: {result['algorithm']}"
        )

    def test_metrics_contain_required_columns(self):
        """Metrics DataFrame must contain k, kmedians_l1_objective, silhouette_score."""
        X = _two_cluster_data(seed=13)
        result = evaluate_kmedians(X, [2, 3])
        required = {"k", "kmedians_l1_objective", "silhouette_score"}
        assert required.issubset(result["metrics"].columns), (
            f"Missing columns: {required - set(result['metrics'].columns)}"
        )

    def test_k_values_match_requested(self):
        """labels_by_k keys must match the requested k values."""
        X = _two_cluster_data(seed=14)
        result = evaluate_kmedians(X, [2, 3, 4])
        assert set(result["labels_by_k"]) == {2, 3, 4}

    def test_silhouette_score_range(self):
        """Silhouette scores must be in [-1, 1]."""
        X = _two_cluster_data(seed=15)
        result = evaluate_kmedians(X, [2, 3])
        for score in result["metrics"]["silhouette_score"].dropna():
            assert -1.0 <= score <= 1.0, f"Silhouette out of range: {score}"

    def test_invalid_k_raises_valueerror(self):
        """k values that are all invalid (k < 2 or k >= n_samples) must raise ValueError."""
        X = np.ones((5, 2))
        with pytest.raises(ValueError):
            evaluate_kmedians(X, [10, 20])

    def test_labels_length_matches_n_samples(self):
        """Each label array must have n_samples entries."""
        X = _two_cluster_data(seed=16)
        result = evaluate_kmedians(X, [2, 3])
        for k, lbls in result["labels_by_k"].items():
            assert len(lbls) == len(X), f"Labels for k={k} have length {len(lbls)}, expected {len(X)}."


# ── compute_cluster_profiles tests ───────────────────────────────────────────

class TestComputeClusterProfiles:
    """Unit tests for the distributional cluster profile function."""

    def _sample_raw_df(self) -> tuple[pd.DataFrame, list[str], np.ndarray]:
        df = pd.DataFrame({
            "country": ["A", "B", "C", "D", "E", "F"],
            "rate_a": [10.0, 12.0, 100.0, 110.0, 200.0, 210.0],
            "rate_b": [1.0, 2.0, 50.0, 60.0, 100.0, 120.0],
        })
        features = ["rate_a", "rate_b"]
        labels = np.array([0, 0, 1, 1, 2, 2])
        return df, features, labels

    def test_returns_dataframe_with_correct_columns(self):
        df, features, labels = self._sample_raw_df()
        profile = compute_cluster_profiles(df, features, labels)
        required_cols = {"cluster", "feature", "country_count", "mean", "median", "std", "q1", "q3", "iqr"}
        assert required_cols.issubset(profile.columns), (
            f"Missing columns: {required_cols - set(profile.columns)}"
        )

    def test_correct_number_of_rows(self):
        """Output must have one row per (cluster, feature) combination."""
        df, features, labels = self._sample_raw_df()
        profile = compute_cluster_profiles(df, features, labels)
        n_clusters = len(np.unique(labels))
        expected_rows = n_clusters * len(features)
        assert len(profile) == expected_rows, (
            f"Expected {expected_rows} rows, got {len(profile)}."
        )

    def test_mean_values_are_correct(self):
        df, features, labels = self._sample_raw_df()
        profile = compute_cluster_profiles(df, features, labels)
        # Cluster 0 countries A and B: rate_a mean = (10+12)/2 = 11.0
        c0_rate_a = profile[(profile["cluster"] == 0) & (profile["feature"] == "rate_a")]["mean"].iloc[0]
        assert abs(c0_rate_a - 11.0) < 1e-6, f"Expected mean 11.0, got {c0_rate_a}."

    def test_median_values_are_correct(self):
        df, features, labels = self._sample_raw_df()
        profile = compute_cluster_profiles(df, features, labels)
        # Cluster 1 countries C and D: rate_b median = (50+60)/2 = 55.0
        c1_rate_b = profile[(profile["cluster"] == 1) & (profile["feature"] == "rate_b")]["median"].iloc[0]
        assert abs(c1_rate_b - 55.0) < 1e-6, f"Expected median 55.0, got {c1_rate_b}."

    def test_iqr_equals_q3_minus_q1(self):
        df, features, labels = self._sample_raw_df()
        profile = compute_cluster_profiles(df, features, labels)
        for _, row in profile.iterrows():
            expected_iqr = round(row["q3"] - row["q1"], 4)
            assert abs(row["iqr"] - expected_iqr) < 1e-6, (
                f"IQR mismatch for cluster={row['cluster']}, feature={row['feature']}."
            )

    def test_raises_on_label_length_mismatch(self):
        df, features, _ = self._sample_raw_df()
        bad_labels = np.array([0, 1, 0])  # length 3, not 6
        with pytest.raises(ValueError, match="does not match labels count"):
            compute_cluster_profiles(df, features, bad_labels)

    def test_country_count_per_cluster(self):
        df, features, labels = self._sample_raw_df()
        profile = compute_cluster_profiles(df, features, labels)
        # All clusters have 2 countries
        for cluster_id in [0, 1, 2]:
            counts = profile[profile["cluster"] == cluster_id]["country_count"].unique()
            assert list(counts) == [2], f"Cluster {cluster_id} country_count unexpected: {counts}."


# ── make_evaluation_curves tests ─────────────────────────────────────────────

class TestMakeEvaluationCurves:
    """Unit tests for the evaluation curves Plotly figure builder."""

    def _kmeans_metrics(self) -> pd.DataFrame:
        return pd.DataFrame({
            "k": [2, 3, 4],
            "wcss_inertia": [100.0, 60.0, 40.0],
            "silhouette_score": [0.7, 0.55, 0.45],
            "davies_bouldin_index": [0.3, 0.5, 0.7],
        })

    def _kmedians_metrics(self) -> pd.DataFrame:
        return pd.DataFrame({
            "k": [2, 3, 4],
            "kmedians_l1_objective": [200.0, 120.0, 80.0],
            "silhouette_score": [0.65, 0.50, 0.40],
        })

    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go
        fig = make_evaluation_curves(self._kmeans_metrics(), algorithm="K-Means (Euclidean)")
        assert isinstance(fig, go.Figure), "Expected a Plotly Figure."

    def test_kmeans_has_three_panels(self):
        """K-Means metrics include DB index, so three panels should be present."""
        fig = make_evaluation_curves(self._kmeans_metrics(), algorithm="K-Means (Euclidean)")
        assert len(fig.data) == 3, f"Expected 3 traces for K-Means, got {len(fig.data)}."

    def test_kmedians_has_two_panels(self):
        """K-Medians metrics omit DB index, so two panels should be present."""
        fig = make_evaluation_curves(self._kmedians_metrics(), algorithm="K-Medians (Manhattan)")
        assert len(fig.data) == 2, f"Expected 2 traces for K-Medians, got {len(fig.data)}."

    def test_x_axis_values_match_k(self):
        """The first trace x-values must equal the k column."""
        fig = make_evaluation_curves(self._kmeans_metrics())
        assert list(fig.data[0].x) == [2, 3, 4]

    def test_title_contains_algorithm_name(self):
        """Figure title must include the provided algorithm name."""
        fig = make_evaluation_curves(self._kmeans_metrics(), algorithm="TestAlgo")
        assert "TestAlgo" in fig.layout.title.text


# ── make_pca_loadings_figure tests ───────────────────────────────────────────

class TestMakePcaLoadingsFigure:
    """Unit tests for the PCA loadings bar chart."""

    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go
        X = _two_cluster_data(seed=20)
        fig = make_pca_loadings_figure(X, ["f1", "f2", "f3"])
        assert isinstance(fig, go.Figure)

    def test_trace_count_equals_n_components(self):
        """Each principal component produces one bar trace."""
        X = _two_cluster_data(seed=21)
        fig = make_pca_loadings_figure(X, ["f1", "f2", "f3"])
        # Two components (PC1 and PC2)
        assert len(fig.data) == 2, f"Expected 2 bar traces, got {len(fig.data)}."

    def test_single_component_when_one_feature(self):
        """With 1 feature PCA can only produce 1 component."""
        X = np.random.default_rng(22).normal(size=(30, 1))
        fig = make_pca_loadings_figure(X, ["only_feature"])
        assert len(fig.data) == 1


# ── Integration: evaluate_kmeans × evaluate_kmedians comparison ───────────────

def test_kmeans_and_kmedians_agree_on_well_separated_data():
    """Both algorithms must produce two clusters on clearly separable data."""
    X = _two_cluster_data(seed=30)
    km_result = evaluate_kmeans(X, [2])
    kmd_result = evaluate_kmedians(X, [2])

    km_labels = km_result["labels_by_k"][2]
    kmd_labels = kmd_result["labels_by_k"][2]

    # Both must produce exactly 2 distinct labels
    assert len(np.unique(km_labels)) == 2
    assert len(np.unique(kmd_labels)) == 2

    # Both must assign the first and second halves to the same label (may be flipped)
    # A correct solution groups the two blobs distinctly
    assert len(np.unique(km_labels[:20])) == 1
    assert len(np.unique(kmd_labels[:20])) == 1
