from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


class KMedians:
    """K-Medians clustering using Manhattan (L1) distance and feature-wise medians.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations for a single run.
    n_init : int, default=10
        Number of time the k-medians algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of minimum L1 objective.
    random_state : int, default=42
        Determines random number generation for centroid initialization.
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        n_init: int = 10,
        random_state: int = 42,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.random_state = int(random_state)
        self.labels_: np.ndarray | None = None
        self.cluster_medians_: np.ndarray | None = None
        self.l1_objective_: float | None = None
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray) -> KMedians:
        self.fit_predict(X)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters}.")
        if self.n_clusters >= n_samples:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) must be strictly less than number of samples ({n_samples})."
            )

        rng = np.random.default_rng(self.random_state)
        best_objective = float("inf")
        best_labels = None
        best_medians = None
        best_n_iter = 0

        for _ in range(self.n_init):
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            init_rng = np.random.default_rng(seed)

            # Sample distinct observations as initial medians
            init_indices = init_rng.choice(n_samples, size=self.n_clusters, replace=False)
            medians = X[init_indices].copy()

            prev_labels = np.full(n_samples, -1, dtype=int)
            labels = np.zeros(n_samples, dtype=int)
            n_iter = 0

            for it in range(self.max_iter):
                n_iter = it + 1
                # 1. Compute Manhattan (L1) distance: shape (n_samples, n_clusters)
                # d(x_i, c_j) = sum_m |x_{i,m} - c_{j,m}|
                diff = np.abs(X[:, np.newaxis, :] - medians[np.newaxis, :, :])
                dist_matrix = np.sum(diff, axis=2)

                # 2. Assignment step
                labels = np.argmin(dist_matrix, axis=1)

                # 3. Check for empty clusters and handle them
                unique_clusters = set(labels)
                if len(unique_clusters) < self.n_clusters:
                    empty_clusters = [c for c in range(self.n_clusters) if c not in unique_clusters]
                    # Assign the points furthest from their assigned medians to the empty clusters
                    min_dists = np.min(dist_matrix, axis=1)
                    sorted_furthest = np.argsort(-min_dists)
                    for rank, empty_c in enumerate(empty_clusters):
                        furthest_idx = sorted_furthest[rank]
                        labels[furthest_idx] = empty_c

                # Check label convergence
                if np.array_equal(labels, prev_labels):
                    break
                prev_labels = labels.copy()

                # 4. Update step: compute component-wise medians
                new_medians = np.zeros_like(medians)
                for c in range(self.n_clusters):
                    cluster_pts = X[labels == c]
                    if len(cluster_pts) > 0:
                        new_medians[c] = np.median(cluster_pts, axis=0)
                    else:
                        new_medians[c] = medians[c]
                medians = new_medians

            # Calculate L1 objective: sum of Manhattan distances to assigned medians
            assigned_medians = medians[labels]
            run_objective = float(np.sum(np.abs(X - assigned_medians)))

            if run_objective < best_objective:
                best_objective = run_objective
                best_labels = labels
                best_medians = medians
                best_n_iter = n_iter

        self.labels_ = best_labels
        self.cluster_medians_ = best_medians
        self.l1_objective_ = best_objective
        self.n_iter_ = best_n_iter
        return self.labels_


def evaluate_kmeans(X: np.ndarray, k_values: list[int], random_state: int = 42) -> dict:
    """Evaluate Euclidean K-Means across specified k cluster counts."""
    n_samples = X.shape[0]
    valid_k = sorted(set(int(k) for k in k_values if 2 <= int(k) < n_samples))
    if not valid_k:
        raise ValueError("Choose k values with 2 <= k < number of usable rows.")

    rows = []
    labels_by_k = {}
    models = {}
    for k in valid_k:
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(X)
        labels_by_k[k] = labels
        models[k] = model

        unique_labels = np.unique(labels)
        try:
            silhouette = (
                float(silhouette_score(X, labels, metric="euclidean"))
                if 2 <= len(unique_labels) < n_samples else np.nan
            )
        except Exception:
            silhouette = np.nan

        try:
            db_index = (
                float(davies_bouldin_score(X, labels))
                if 2 <= len(unique_labels) < n_samples else np.nan
            )
        except Exception:
            db_index = np.nan
        rows.append({
            "k": k,
            "wcss_inertia": float(model.inertia_),
            "silhouette_score": silhouette,
            "davies_bouldin_index": db_index,
            "clusters_observed": int(len(unique_labels)),
        })

    return {
        "algorithm": "K-Means (Euclidean)",
        "metrics": pd.DataFrame(rows).sort_values("k").reset_index(drop=True),
        "labels_by_k": labels_by_k,
        "models": models,
    }


def evaluate_kmedians(X: np.ndarray, k_values: list[int], random_state: int = 42) -> dict:
    """Evaluate Manhattan K-Medians across specified k cluster counts."""
    n_samples = X.shape[0]
    valid_k = sorted(set(int(k) for k in k_values if 2 <= int(k) < n_samples))
    if not valid_k:
        raise ValueError("Choose k values with 2 <= k < number of usable rows.")

    rows = []
    labels_by_k = {}
    models = {}
    for k in valid_k:
        model = KMedians(n_clusters=k, n_init=15, max_iter=300, random_state=random_state)
        labels = model.fit_predict(X)
        labels_by_k[k] = labels
        models[k] = model

        unique_labels = np.unique(labels)
        try:
            silhouette = (
                float(silhouette_score(X, labels, metric="manhattan"))
                if 2 <= len(unique_labels) < n_samples else np.nan
            )
        except Exception:
            silhouette = np.nan
        rows.append({
            "k": k,
            "kmedians_l1_objective": float(model.l1_objective_),
            "silhouette_score": silhouette,
            "clusters_observed": int(len(unique_labels)),
        })

    return {
        "algorithm": "K-Medians (Manhattan)",
        "metrics": pd.DataFrame(rows).sort_values("k").reset_index(drop=True),
        "labels_by_k": labels_by_k,
        "models": models,
    }


def compute_cluster_profiles(
    raw_df: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    """Calculate distributional profile statistics (mean, median, std, Q1, Q3, IQR)

    Statistics are computed on the unscaled/original feature values for clear
    public-health interpretability.
    """
    if len(raw_df) != len(labels):
        raise ValueError(
            f"Row count of DataFrame ({len(raw_df)}) does not match labels count ({len(labels)})."
        )

    records = []
    clusters = sorted(np.unique(labels))

    for c in clusters:
        cluster_mask = (labels == c)
        c_subset = raw_df.loc[cluster_mask]
        c_count = int(cluster_mask.sum())

        for feat in features:
            series = pd.to_numeric(c_subset[feat], errors="coerce").dropna()
            if series.empty:
                records.append({
                    "cluster": int(c),
                    "feature": feat,
                    "country_count": c_count,
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                    "iqr": np.nan,
                })
                continue

            q1 = float(np.percentile(series, 25))
            q3 = float(np.percentile(series, 75))
            records.append({
                "cluster": int(c),
                "feature": feat,
                "country_count": c_count,
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std(ddof=1)), 4) if len(series) > 1 else 0.0,
                "q1": round(q1, 4),
                "q3": round(q3, 4),
                "iqr": round(q3 - q1, 4),
            })

    return pd.DataFrame(records)
