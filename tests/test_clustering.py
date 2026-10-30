import numpy as np
from src.clustering import evaluate_kmeans


def test_evaluate_multiple_k_values():
    rng = np.random.default_rng(7)
    X = np.vstack([
        rng.normal(loc=-3, scale=0.3, size=(20, 3)),
        rng.normal(loc=3, scale=0.3, size=(20, 3)),
    ])
    result = evaluate_kmeans(X, [2, 3, 4])
    assert list(result["metrics"]["k"]) == [2, 3, 4]
    assert {"wcss_inertia", "silhouette_score", "davies_bouldin_index"}.issubset(
        result["metrics"].columns
    )
    assert set(result["labels_by_k"]) == {2, 3, 4}
