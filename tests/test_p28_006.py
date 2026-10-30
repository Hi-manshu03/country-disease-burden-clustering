"""Tests for P28-006: PCA explained variance, scree plot, and loading interpretation.

All tests use synthetic data and run without internet access.
No test depends on exact pixel-level visual output; they test numerical
correctness and chart structure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from src.visualization import (
    fit_pca,
    make_pca_figure,
    make_pca_loadings_figure,
    make_scree_figure,
)
from src.data import prepare_features


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _scaled_matrix(n_samples: int = 40, n_features: int = 5, seed: int = 0) -> np.ndarray:
    """Return a StandardScaler-normalised random matrix (simulates prepare_features output)."""
    rng = np.random.default_rng(seed)
    X_raw = rng.normal(size=(n_samples, n_features))
    # Mimic StandardScaler (mean=0, std=1 per column)
    X_raw -= X_raw.mean(axis=0)
    X_raw /= X_raw.std(axis=0, ddof=0) + 1e-12
    return X_raw


def _two_cluster_scaled(seed: int = 42) -> np.ndarray:
    """Return a 40×3 standardised matrix with two well-separated clusters."""
    rng = np.random.default_rng(seed)
    X = np.vstack([
        rng.normal(-3, 0.4, (20, 3)),
        rng.normal(3, 0.4, (20, 3)),
    ])
    X -= X.mean(axis=0)
    X /= X.std(axis=0, ddof=0) + 1e-12
    return X


# ── Tests: fit_pca ────────────────────────────────────────────────────────────

class TestFitPca:
    """Unit tests for the fit_pca helper."""

    def test_returns_pca_model_and_coords(self):
        X = _scaled_matrix(40, 5)
        pca, coords = fit_pca(X)
        assert isinstance(pca, PCA)
        assert coords.shape[0] == 40

    def test_default_n_components_is_min_samples_features(self):
        """Without n_components, PCA should compute min(n_samples, n_features) components."""
        X = _scaled_matrix(40, 5)
        pca, coords = fit_pca(X)
        expected = min(40, 5)
        assert pca.n_components_ == expected
        assert coords.shape[1] == expected

    def test_explicit_n_components_respected(self):
        X = _scaled_matrix(40, 5)
        pca, coords = fit_pca(X, n_components=2)
        assert pca.n_components_ == 2
        assert coords.shape == (40, 2)

    def test_n_components_clamped_to_max(self):
        """Requesting more components than min(n_samples, n_features) should clamp silently."""
        X = _scaled_matrix(10, 3)
        pca, coords = fit_pca(X, n_components=100)
        assert pca.n_components_ <= min(10, 3)

    def test_raises_on_single_sample(self):
        X = _scaled_matrix(1, 3)
        with pytest.raises(ValueError, match="at least 2 samples"):
            fit_pca(X)

    def test_raises_on_single_feature(self):
        X = _scaled_matrix(10, 1)
        with pytest.raises(ValueError, match="at least 2 features"):
            fit_pca(X)

    def test_explained_variance_ratios_sum_to_one(self):
        """Full PCA must account for all variance (sum ≈ 1.0)."""
        X = _scaled_matrix(30, 4, seed=1)
        pca, _ = fit_pca(X, n_components=None)
        assert abs(pca.explained_variance_ratio_.sum() - 1.0) < 1e-6

    def test_explained_variance_ratios_are_non_negative(self):
        X = _scaled_matrix(40, 6, seed=2)
        pca, _ = fit_pca(X)
        assert np.all(pca.explained_variance_ratio_ >= 0)

    def test_explained_variance_ratios_are_descending(self):
        """Explained variance must be non-increasing across components."""
        X = _scaled_matrix(40, 5, seed=3)
        pca, _ = fit_pca(X)
        ratios = pca.explained_variance_ratio_
        assert np.all(ratios[:-1] >= ratios[1:] - 1e-10), (
            f"Variance ratios are not descending: {ratios}"
        )

    def test_coords_shape_matches_n_samples(self):
        X = _scaled_matrix(50, 7, seed=4)
        pca, coords = fit_pca(X, n_components=3)
        assert coords.shape == (50, 3)

    def test_determinism_with_fixed_random_state(self):
        """Two calls with the same data must produce identical results."""
        X = _scaled_matrix(40, 5, seed=5)
        pca1, coords1 = fit_pca(X)
        pca2, coords2 = fit_pca(X)
        np.testing.assert_allclose(pca1.explained_variance_ratio_, pca2.explained_variance_ratio_)
        np.testing.assert_allclose(np.abs(coords1), np.abs(coords2), atol=1e-10)

    def test_known_variance_on_identity_covariance(self):
        """For orthogonal data with equal variance, each component should capture 1/n_features."""
        n_features = 4
        X = np.array([
            [1.0 if (mask >> j) & 1 else -1.0
             for j in range(n_features)]
            for mask in range(2 ** n_features)
        ])
        X -= X.mean(axis=0)
        X /= X.std(axis=0, ddof=0) + 1e-12
        
        pca, _ = fit_pca(X, n_components=n_features)
        # With large sample size and equal variance, each ratio should be ≈ 1/n_features
        expected = 1.0 / n_features
        np.testing.assert_allclose(
            pca.explained_variance_ratio_, expected, atol=1e-6,
            err_msg="Equal-variance orthogonal data: each component should capture ≈1/n_features."
        )


# ── Tests: make_scree_figure ──────────────────────────────────────────────────

class TestMakeScreeFigure:
    """Tests for the scree plot figure builder."""

    def _fitted_pca(self, n_comp: int = 4) -> PCA:
        X = _scaled_matrix(40, n_comp + 1, seed=10)
        pca, _ = fit_pca(X, n_components=n_comp)
        return pca

    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go
        pca = self._fitted_pca()
        fig = make_scree_figure(pca)
        assert isinstance(fig, go.Figure)

    def test_has_two_panels(self):
        """Scree figure must have exactly 2 subplots: bar + cumulative line."""
        pca = self._fitted_pca()
        fig = make_scree_figure(pca)
        # Bar trace (individual) + Scatter trace (cumulative) = 2 data traces
        data_traces = [t for t in fig.data if not isinstance(t, type(None))]
        assert len(data_traces) == 2, f"Expected 2 data traces, got {len(data_traces)}."

    def test_bar_y_values_match_individual_variance(self):
        """Bar chart y-values must match pca.explained_variance_ratio_ * 100."""
        pca = self._fitted_pca(n_comp=3)
        fig = make_scree_figure(pca)
        bar_trace = next(t for t in fig.data if hasattr(t, "y") and hasattr(t, "type") and t.type == "bar")
        expected = (pca.explained_variance_ratio_ * 100).tolist()
        np.testing.assert_allclose(list(bar_trace.y), expected, atol=1e-6)

    def test_cumulative_trace_y_values_correct(self):
        """Cumulative scatter trace must equal cumsum of explained_variance_ratio_ * 100."""
        pca = self._fitted_pca(n_comp=4)
        fig = make_scree_figure(pca)
        scatter_trace = next(t for t in fig.data if t.type == "scatter")
        expected = (np.cumsum(pca.explained_variance_ratio_) * 100).tolist()
        np.testing.assert_allclose(list(scatter_trace.y), expected, atol=1e-6)

    def test_x_labels_are_pc_prefixed(self):
        """Bar chart x-labels must be 'PC1', 'PC2', etc."""
        pca = self._fitted_pca(n_comp=3)
        fig = make_scree_figure(pca)
        bar_trace = next(t for t in fig.data if t.type == "bar")
        labels = list(bar_trace.x)
        assert labels == ["PC1", "PC2", "PC3"], f"Unexpected labels: {labels}"

    def test_last_cumulative_value_is_100_pct(self):
        """Last cumulative variance value must equal total variance (≈100%)."""
        X = _scaled_matrix(30, 4, seed=11)
        pca, _ = fit_pca(X, n_components=None)
        fig = make_scree_figure(pca)
        scatter_trace = next(t for t in fig.data if t.type == "scatter")
        assert abs(scatter_trace.y[-1] - 100.0) < 1e-4, (
            f"Last cumulative value should be ≈100%, got {scatter_trace.y[-1]:.4f}."
        )

    def test_title_contains_scree(self):
        pca = self._fitted_pca()
        fig = make_scree_figure(pca)
        assert "Scree" in fig.layout.title.text or "scree" in fig.layout.title.text.lower()

    def test_works_with_two_component_pca(self):
        """Scree plot must work when only 2 components are available."""
        X = _scaled_matrix(20, 2, seed=12)
        pca, _ = fit_pca(X, n_components=2)
        fig = make_scree_figure(pca)
        bar_trace = next(t for t in fig.data if t.type == "bar")
        assert len(bar_trace.y) == 2


# ── Tests: make_pca_figure (with pca_model param) ────────────────────────────

class TestMakePcaFigureWithModel:
    """Tests for make_pca_figure when a pre-fitted pca_model is supplied."""

    def test_accepts_prefitted_model(self):
        """make_pca_figure should work without error when given a pre-fitted model."""
        import plotly.graph_objects as go
        X = _two_cluster_scaled()
        pca, _ = fit_pca(X, n_components=None)
        labels = np.array([0] * 20 + [1] * 20)
        countries = [f"C{i}" for i in range(40)]
        fig = make_pca_figure(X, labels, countries, pca_model=pca)
        assert isinstance(fig, go.Figure)

    def test_pca_model_coordinates_consistent_with_full_fit(self):
        """PC1/PC2 coords from make_pca_figure(pca_model=...) must match pca.transform(X)[:, :2]."""
        X = _two_cluster_scaled()
        pca, _ = fit_pca(X, n_components=None)
        labels = np.zeros(40, dtype=int)
        countries = [f"C{i}" for i in range(40)]
        fig = make_pca_figure(X, labels, countries, pca_model=pca)

        # Extract x-axis data from the figure's first trace
        fig_pc1 = np.array(fig.data[0].x)
        expected_pc1 = pca.transform(X)[:, 0]
        # Signs can be globally flipped; compare absolute values
        np.testing.assert_allclose(
            np.abs(fig_pc1), np.abs(expected_pc1), atol=1e-8,
            err_msg="PC1 coordinates in figure do not match pca.transform(X)[:, 0]."
        )

    def test_axis_labels_contain_variance_pct(self):
        """PC1 and PC2 axis labels must contain the variance percentage from the model."""
        X = _two_cluster_scaled()
        pca, _ = fit_pca(X, n_components=None)
        labels = np.zeros(40, dtype=int)
        countries = [f"C{i}" for i in range(40)]
        fig = make_pca_figure(X, labels, countries, pca_model=pca)
        xaxis_title = fig.layout.xaxis.title.text
        assert "%" in xaxis_title, f"PC1 axis title should contain '%': {xaxis_title}"
        assert "PC1" in xaxis_title, f"PC1 axis title should contain 'PC1': {xaxis_title}"

    def test_scatter_coords_align_with_country_names(self):
        """The scatter plot must have one point per country, in the correct order."""
        X = _two_cluster_scaled()
        pca, _ = fit_pca(X, n_components=None)
        labels = np.zeros(40, dtype=int)
        countries = [f"Country_{i}" for i in range(40)]
        fig = make_pca_figure(X, labels, countries, pca_model=pca)
        # hover_name is stored as customdata or text depending on Plotly version;
        # check that the number of points matches number of countries
        total_points = sum(len(t.x) for t in fig.data if hasattr(t, "x") and t.x is not None)
        assert total_points == 40, f"Expected 40 scatter points, got {total_points}."

    def test_raises_when_country_names_length_mismatch(self):
        X = _two_cluster_scaled()
        pca, _ = fit_pca(X, n_components=None)
        labels = np.zeros(40, dtype=int)
        with pytest.raises(ValueError, match="does not match cluster labels count"):
            make_pca_figure(X, labels, ["OnlyOne"], pca_model=pca)

    def test_backward_compat_without_pca_model(self):
        """make_pca_figure without pca_model must still work (no regression)."""
        import plotly.graph_objects as go
        X = _two_cluster_scaled()
        labels = np.array([0] * 20 + [1] * 20)
        countries = [f"C{i}" for i in range(40)]
        fig = make_pca_figure(X, labels, countries)
        assert isinstance(fig, go.Figure)


# ── Tests: make_pca_loadings_figure ──────────────────────────────────────────

class TestMakePcaLoadingsFigure:
    """Tests for the updated loadings figure with pca_model support and heatmap path."""

    def test_bar_chart_for_two_components(self):
        """With 2 components the function must return a bar chart (n traces == 2)."""
        X = _scaled_matrix(30, 4, seed=20)
        pca, _ = fit_pca(X, n_components=2)
        fig = make_pca_loadings_figure(X, ["f1", "f2", "f3", "f4"], pca_model=pca)
        # Bar chart path: one trace per component
        assert len(fig.data) == 2, f"Expected 2 bar traces, got {len(fig.data)}."

    def test_heatmap_for_three_plus_components(self):
        """With 3+ components the function must return a heatmap (single Heatmap trace)."""
        import plotly.graph_objects as go
        X = _scaled_matrix(30, 5, seed=21)
        pca, _ = fit_pca(X, n_components=5)
        fig = make_pca_loadings_figure(X, ["f1", "f2", "f3", "f4", "f5"], pca_model=pca)
        assert any(isinstance(t, go.Heatmap) for t in fig.data), (
            "Expected a Heatmap trace for 5 components."
        )

    def test_heatmap_z_matches_pca_components(self):
        """Heatmap z-values must equal pca.components_.T (features × components)."""
        X = _scaled_matrix(30, 4, seed=22)
        feats = ["f1", "f2", "f3", "f4"]
        pca, _ = fit_pca(X, n_components=4)
        fig = make_pca_loadings_figure(X, feats, pca_model=pca)
        import plotly.graph_objects as go
        heatmap = next(t for t in fig.data if isinstance(t, go.Heatmap))
        z_fig = np.array(heatmap.z)  # shape (n_features, n_components)
        z_expected = pca.components_.T  # shape (n_features, n_components)
        np.testing.assert_allclose(z_fig, z_expected, atol=1e-10,
                                   err_msg="Heatmap z-values must equal pca.components_.T")

    def test_heatmap_y_labels_are_feature_names(self):
        """Heatmap row labels (y) must match the provided feature names."""
        import plotly.graph_objects as go
        X = _scaled_matrix(30, 3, seed=23)
        feats = ["alpha", "beta", "gamma"]
        pca, _ = fit_pca(X, n_components=3)
        fig = make_pca_loadings_figure(X, feats, pca_model=pca)
        heatmap = next(t for t in fig.data if isinstance(t, go.Heatmap))
        assert list(heatmap.y) == feats, f"Y-labels mismatch: {list(heatmap.y)} != {feats}"

    def test_heatmap_x_labels_contain_pc_prefix_and_pct(self):
        """Heatmap column labels (x) must contain 'PC' and '%'."""
        import plotly.graph_objects as go
        X = _scaled_matrix(30, 4, seed=24)
        pca, _ = fit_pca(X, n_components=4)
        fig = make_pca_loadings_figure(X, ["f1", "f2", "f3", "f4"], pca_model=pca)
        heatmap = next(t for t in fig.data if isinstance(t, go.Heatmap))
        for label in heatmap.x:
            assert "PC" in label and "%" in label, f"Bad component label: {label}"

    def test_max_components_heatmap_limits_columns(self):
        """max_components_heatmap must limit the number of displayed heatmap columns."""
        import plotly.graph_objects as go
        X = _scaled_matrix(30, 6, seed=25)
        pca, _ = fit_pca(X, n_components=6)
        fig = make_pca_loadings_figure(
            X, [f"f{i}" for i in range(6)], pca_model=pca, max_components_heatmap=3
        )
        heatmap = next(t for t in fig.data if isinstance(t, go.Heatmap))
        assert len(heatmap.x) == 3, f"Expected 3 heatmap columns, got {len(heatmap.x)}."

    def test_backward_compat_without_pca_model(self):
        """make_pca_loadings_figure must still work when pca_model=None."""
        X = _scaled_matrix(30, 3, seed=26)
        fig = make_pca_loadings_figure(X, ["f1", "f2", "f3"])
        assert len(fig.data) >= 1

    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go
        X = _scaled_matrix(30, 4, seed=27)
        pca, _ = fit_pca(X, n_components=4)
        fig = make_pca_loadings_figure(X, ["f1", "f2", "f3", "f4"], pca_model=pca)
        assert isinstance(fig, go.Figure)


# ── Tests: PCA preprocessing correctness ─────────────────────────────────────

class TestPcaPreprocessingCorrectness:
    """Verify that PCA uses the scaled feature matrix from prepare_features."""

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "country": [f"C{i}" for i in range(20)],
            "rate_a": np.linspace(1, 20, 20),
            "rate_b": np.linspace(100, 1000, 20),
            "rate_c": np.random.default_rng(0).normal(50, 10, 20),
        })

    def test_prepare_features_output_is_standardised(self):
        """X from prepare_features should have approximately zero mean and unit variance per column."""
        df = self._make_df()
        prepared = prepare_features(df, ["rate_a", "rate_b", "rate_c"])
        X = prepared["X"]
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(X.std(axis=0, ddof=0), 1, atol=1e-10)

    def test_pca_on_scaled_matrix_is_consistent(self):
        """fit_pca on standardised matrix must produce the same variance ratios as
        manually fitting PCA on the same matrix."""
        df = self._make_df()
        prepared = prepare_features(df, ["rate_a", "rate_b", "rate_c"])
        X = prepared["X"]
        pca_via_helper, _ = fit_pca(X, n_components=None)
        pca_manual = PCA(n_components=min(X.shape), random_state=42)
        pca_manual.fit(X)
        np.testing.assert_allclose(
            pca_via_helper.explained_variance_ratio_,
            pca_manual.explained_variance_ratio_,
            atol=1e-10,
        )

    def test_country_coord_cluster_alignment(self):
        """PCA coordinates must be aligned with country names and cluster labels."""
        df = self._make_df()
        prepared = prepare_features(df, ["rate_a", "rate_b", "rate_c"])
        X = prepared["X"]
        n = X.shape[0]
        labels = np.zeros(n, dtype=int)
        countries = [f"C{i}" for i in range(n)]
        pca, coords = fit_pca(X, n_components=None)

        assert len(countries) == len(labels) == n
        assert coords.shape[0] == n

        fig = make_pca_figure(X, labels, countries, pca_model=pca)
        total_points = sum(len(t.x) for t in fig.data if hasattr(t, "x") and t.x is not None)
        assert total_points == n

    def test_pca_model_reuse_gives_same_variance(self):
        """Using pca_model in make_pca_figure must give same PC1 variance % as scree plot."""
        df = self._make_df()
        prepared = prepare_features(df, ["rate_a", "rate_b", "rate_c"])
        X = prepared["X"]
        pca, _ = fit_pca(X, n_components=None)

        # Variance from scree plot
        scree_fig = make_scree_figure(pca)
        bar_trace = next(t for t in scree_fig.data if t.type == "bar")
        pc1_var_scree = bar_trace.y[0]

        # Variance from PCA figure axis label
        labels = np.zeros(X.shape[0], dtype=int)
        countries = [f"C{i}" for i in range(X.shape[0])]
        scatter_fig = make_pca_figure(X, labels, countries, pca_model=pca)
        xaxis_title = scatter_fig.layout.xaxis.title.text
        # Extract the percentage from label like "PC1 (38.4%)"
        import re
        match = re.search(r"(\d+\.\d+)%", xaxis_title)
        assert match is not None, f"Could not parse % from axis label: {xaxis_title}"
        pc1_var_scatter = float(match.group(1))

        assert abs(pc1_var_scree - pc1_var_scatter) < 0.05, (
            f"PC1 variance mismatch between scree ({pc1_var_scree:.2f}%) "
            f"and scatter axis ({pc1_var_scatter:.2f}%)."
        )


# ── Tests: edge cases and degenerate inputs ───────────────────────────────────

class TestPcaEdgeCases:
    """Edge-case and degenerate input handling."""

    def test_minimum_valid_dataset(self):
        """3 samples, 2 features should produce 2 components without error."""
        X = np.array([[1.0, -1.0], [-1.0, 1.0], [0.5, -0.5]])
        X -= X.mean(axis=0)
        std = X.std(axis=0, ddof=0) + 1e-12
        X /= std
        pca, coords = fit_pca(X)
        assert pca.n_components_ == 2
        fig = make_scree_figure(pca)
        assert fig is not None

    def test_square_matrix(self):
        """n_samples == n_features: max components = n_features."""
        X = _scaled_matrix(5, 5, seed=50)
        pca, coords = fit_pca(X)
        assert pca.n_components_ == 5

    def test_more_features_than_samples(self):
        """n_features > n_samples: max components = n_samples."""
        X = _scaled_matrix(8, 20, seed=51)
        pca, coords = fit_pca(X)
        assert pca.n_components_ == 8
        assert coords.shape == (8, 8)

    def test_scree_plot_single_component_model(self):
        """Scree plot with a 1-component PCA must produce a figure without error."""
        X = _scaled_matrix(20, 4, seed=52)
        pca, _ = fit_pca(X, n_components=1)
        fig = make_scree_figure(pca)
        assert fig is not None

    def test_loadings_figure_many_features(self):
        """Loadings figure with 15 features must return a valid figure (heatmap path)."""
        n_feat = 15
        X = _scaled_matrix(40, n_feat, seed=53)
        feats = [f"feat_{i}" for i in range(n_feat)]
        pca, _ = fit_pca(X, n_components=n_feat)
        fig = make_pca_loadings_figure(X, feats, pca_model=pca)
        assert fig is not None
        assert len(fig.data) >= 1

    def test_pca_figure_single_cluster(self):
        """All countries in one cluster: PCA scatter must still work."""
        X = _two_cluster_scaled()
        pca, _ = fit_pca(X, n_components=None)
        labels = np.zeros(40, dtype=int)  # single cluster
        countries = [f"C{i}" for i in range(40)]
        fig = make_pca_figure(X, labels, countries, pca_model=pca)
        assert fig is not None
