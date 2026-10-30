from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data import prepare_features, validate_dataset
from src.clustering import evaluate_kmeans, evaluate_kmedians, compute_cluster_profiles
from src.visualization import (
    fit_pca,
    make_pca_figure,
    make_scree_figure,
    make_evaluation_curves,
    make_pca_loadings_figure,
)

st.set_page_config(page_title="Country Disease-Burden Clustering", page_icon="🌍", layout="wide")

st.title("Country Disease-Burden Pattern Clustering")
st.caption("Unsupervised public-health analysis • Project 28")

st.warning(
    "Use verified, documented IHME indicators for final results. "
    "Clusters describe statistical similarities; they do not establish causes or diagnoses."
)

uploaded = st.file_uploader("Upload a prepared country-level CSV", type=["csv"])

if uploaded is None:
    st.info(
        "Expected input: one row per country and numeric disease-burden indicators. "
        "Include a country name/code column and document the year, measure, units, and source."
    )
    st.stop()

try:
    raw = pd.read_csv(uploaded)
    country_col, numeric_cols = validate_dataset(raw)
except Exception as exc:
    st.error(f"Could not read or validate the CSV: {exc}")
    st.stop()

if not numeric_cols:
    st.error("No usable numeric feature columns were found.")
    st.stop()

default_features = numeric_cols[: min(5, len(numeric_cols))]
features = st.multiselect("Select disease-burden features", numeric_cols, default=default_features)
k_values = st.multiselect("Choose cluster counts (select at least three)", [2, 3, 4, 5, 6, 7, 8], default=[2, 3, 4])

# ── Distance metric / algorithm selector ─────────────────────────────────────
algorithm_choice = st.selectbox(
    "Clustering algorithm",
    options=["K-Means (Euclidean)", "K-Medians (Manhattan)"],
    help=(
        "K-Means minimises the sum of squared Euclidean distances (WCSS) and "
        "uses feature means as centroids. "
        "K-Medians minimises the sum of Manhattan (L1) distances and uses feature-wise "
        "medians as centroids; it is more robust to outlying observations."
    ),
    key="algorithm_choice",
)

missing_threshold_pct = st.slider(
    "Maximum allowed missing features per country (%)",
    min_value=0,
    max_value=100,
    value=30,
    step=5,
    help="Countries missing more than this percentage of the selected features will be excluded from clustering.",
)
max_missing_fraction = missing_threshold_pct / 100.0

if not features:
    st.info("Select at least two numeric features to continue.")
    st.stop()
if len(features) < 2:
    st.info("Select at least two numeric features for a meaningful PCA visualization.")
    st.stop()
if len(k_values) < 3:
    st.info("For the assignment requirement, select at least three cluster counts.")
    st.stop()

try:
    prepared = prepare_features(
        raw,
        features,
        max_missing_fraction=max_missing_fraction,
        country_col=country_col,
    )
except Exception as exc:
    st.error(f"Preprocessing failed: {exc}")
    st.stop()

st.subheader("Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Input rows", f"{len(raw):,}")
c2.metric("Retained rows", f"{prepared['retained_rows']:,}")
c3.metric("Excluded rows", f"{prepared['excluded_rows']:,}")
c4.metric("Missing values imputed", f"{prepared['imputed_cells']:,}")

if prepared["excluded_rows"] > 0:
    st.warning(
        f"⚠️ {prepared['excluded_rows']} country observation(s) exceeded the {missing_threshold_pct}% "
        "missingness threshold and were excluded from clustering. "
        "Note: Excluding data-deficient countries can alter the population and geographic balance represented by the analysis."
    )
    with st.expander("Inspect excluded countries and audit log"):
        st.dataframe(prepared["exclusions_df"], use_container_width=True)
        st.download_button(
            "Download exclusion audit (CSV)",
            data=prepared["exclusions_df"].to_csv(index=False).encode("utf-8"),
            file_name="excluded_countries_audit.csv",
            mime="text/csv",
            key="dl_exclusions",
        )

with st.expander("Preview input data"):
    st.dataframe(raw.head(20), use_container_width=True)

file_signature = getattr(uploaded, "file_id", None) or f"{uploaded.name}_{uploaded.size}"

if st.button("Run clustering experiments", type="primary"):
    if prepared["retained_rows"] < 3:
        st.error("At least three valid country rows are needed.")
        st.stop()
    analysis_raw = raw.loc[prepared["row_indices"]].copy()

    # Reset the index of analysis_raw so it aligns with X (which is always 0-indexed)
    analysis_raw = analysis_raw.reset_index(drop=True)

    # ── Fit PCA once on the standardised feature matrix ───────────────────
    # n_components=None → maximum number of components (min(n_samples, n_features))
    # All PCA charts (scatter, scree, loadings) must use this single fitted model
    # so their explained-variance numbers are consistent.
    try:
        pca_model, _pca_coords = fit_pca(prepared["X"], n_components=None)
    except ValueError as exc:
        st.error(f"PCA failed: {exc}")
        st.stop()

    try:
        if algorithm_choice == "K-Means (Euclidean)":
            results = evaluate_kmeans(
                X=prepared["X"],
                k_values=k_values,
                random_state=42,
            )
        else:
            results = evaluate_kmedians(
                X=prepared["X"],
                k_values=k_values,
                random_state=42,
            )
    except ValueError as exc:
        st.error(f"Clustering failed: {exc}")
        st.stop()

    omitted_k = [k for k in k_values if k not in results["labels_by_k"]]
    if omitted_k:
        st.warning(
            f"Note: Cluster count(s) {omitted_k} were omitted because k must be strictly less "
            f"than the number of retained country observations ({prepared['retained_rows']})."
        )

    if (
        "selected_k_solution" in st.session_state
        and st.session_state["selected_k_solution"] not in results["labels_by_k"]
    ):
        st.session_state.pop("selected_k_solution", None)

    st.session_state["clustering_result"] = {
        "algorithm": results["algorithm"],
        "metrics": results["metrics"],
        "labels_by_k": results["labels_by_k"],
        "X": prepared["X"],
        "pca_model": pca_model,       # single fitted model shared by all PCA views
        "analysis_raw": analysis_raw,
        "country_col": country_col,
        "country_names": analysis_raw[country_col].astype(str).tolist(),
        "features": list(features),
        "k_values": list(k_values),
        "max_missing_fraction": max_missing_fraction,
        "algorithm_choice": algorithm_choice,
        "file_signature": file_signature,
    }

if "clustering_result" in st.session_state:
    cached = st.session_state["clustering_result"]
    if (
        cached.get("file_signature") != file_signature
        or cached.get("features") != list(features)
        or cached.get("k_values") != list(k_values)
        or cached.get("max_missing_fraction") != max_missing_fraction
        or cached.get("algorithm_choice") != algorithm_choice
    ):
        st.session_state.pop("clustering_result", None)
        st.session_state.pop("selected_k_solution", None)
    else:
        metrics = cached["metrics"]
        algorithm_label = cached["algorithm"]
        pca_model = cached["pca_model"]

        # ── Evaluation curves ─────────────────────────────────────────────
        st.subheader("Model evaluation")
        eval_fig = make_evaluation_curves(metrics, algorithm=algorithm_label)
        st.plotly_chart(eval_fig, use_container_width=True)

        with st.expander("View evaluation metrics table"):
            st.dataframe(metrics, use_container_width=True)
            st.download_button(
                "Download evaluation metrics (CSV)",
                data=metrics.to_csv(index=False).encode("utf-8"),
                file_name="clustering_evaluation.csv",
                mime="text/csv",
                key="dl_metrics",
            )

        # ── Cluster solution selector ─────────────────────────────────────
        available_k = list(cached["labels_by_k"].keys())
        selected_k = st.selectbox(
            "Cluster solution to inspect",
            available_k,
            key="selected_k_solution",
        )
        labels = cached["labels_by_k"][selected_k]

        # ── PCA scatter ───────────────────────────────────────────────────
        st.subheader("PCA cluster visualization")
        st.caption(
            "Countries are projected onto the first two principal components of the standardised "
            "feature matrix for visualisation only. Clustering was computed on all selected features, "
            "not on these two axes. The percentage on each axis shows how much variance that "
            "component individually captures."
        )
        fig = make_pca_figure(cached["X"], labels, cached["country_names"], pca_model=pca_model)
        st.plotly_chart(fig, use_container_width=True)

        # ── PCA analysis: scree + loadings ────────────────────────────────
        st.subheader("PCA analysis")
        pca_tab_scree, pca_tab_loadings = st.tabs(["Scree plot", "Feature loadings"])

        with pca_tab_scree:
            scree_fig = make_scree_figure(pca_model)
            st.plotly_chart(scree_fig, use_container_width=True)
            n_total = len(pca_model.explained_variance_ratio_)
            cumvar = pca_model.explained_variance_ratio_.cumsum() * 100
            st.caption(
                f"This dataset has {n_total} principal component(s). "
                f"PC1 + PC2 together explain "
                f"{pca_model.explained_variance_ratio_[:2].sum()*100:.1f}% of total variance. "
                "The scree plot helps assess dimensionality but does not determine the number of clusters. "
                "Reference lines mark 90% and 95% cumulative variance."
            )
            # Tabular summary
            with st.expander("Variance summary table"):
                var_rows = []
                for i, vr in enumerate(pca_model.explained_variance_ratio_):
                    var_rows.append({
                        "Component": f"PC{i+1}",
                        "Individual variance (%)": f"{vr*100:.2f}",
                        "Cumulative variance (%)": f"{cumvar[i]:.2f}",
                    })
                st.dataframe(pd.DataFrame(var_rows), use_container_width=True)

        with pca_tab_loadings:
            loadings_fig = make_pca_loadings_figure(
                cached["X"], cached["features"], pca_model=pca_model
            )
            st.plotly_chart(loadings_fig, use_container_width=True)
            st.caption(
                "**What these coefficients mean:** Each value is a linear weight showing how much "
                "a standardised feature contributes to that principal component's direction. "
                "Larger absolute values → stronger contribution. "
                "The sign indicates direction within the component (components can be globally "
                "flipped without changing the PCA solution). "
                "These coefficients are *not* evidence of causality or direct measures of disease burden."
            )

        # ── Distributional cluster profiles ───────────────────────────────
        st.subheader("Cluster profiles")
        st.caption(
            "Statistics computed on the original (unscaled) feature values for public-health interpretability. "
            "Median and IQR are reported alongside mean and SD because disease-burden indicators are often right-skewed."
        )

        profile_df = compute_cluster_profiles(
            raw_df=cached["analysis_raw"],
            features=cached["features"],
            labels=labels,
        )

        profile_tab1, profile_tab2 = st.tabs(["Summary table", "By feature"])

        with profile_tab1:
            st.dataframe(profile_df, use_container_width=True)

        with profile_tab2:
            for feat in cached["features"]:
                feat_data = profile_df[profile_df["feature"] == feat].copy()
                if feat_data.empty:
                    continue
                feat_data = feat_data.sort_values("cluster")
                fig_feat = None
                try:
                    fig_feat = _make_cluster_bar(feat_data, feat)
                except Exception:
                    pass
                if fig_feat is not None:
                    st.plotly_chart(fig_feat, use_container_width=True)

        # ── Country assignments download ──────────────────────────────────
        out = cached["analysis_raw"].copy()
        out["cluster"] = labels
        st.download_button(
            "Download country cluster assignments (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name=f"country_clusters_k{selected_k}.csv",
            mime="text/csv",
            key=f"dl_clusters_k{selected_k}",
        )


def _make_cluster_bar(feat_data: pd.DataFrame, feat_name: str):
    """Return a grouped bar chart showing mean and median per cluster for one feature."""
    import plotly.graph_objects as go

    clusters = feat_data["cluster"].tolist()
    fig = go.Figure()
    fig.add_bar(
        x=clusters,
        y=feat_data["mean"].tolist(),
        name="Mean",
        marker_color="#4F8EF7",
        error_y=dict(type="data", array=feat_data["std"].tolist(), visible=True),
    )
    fig.add_bar(
        x=clusters,
        y=feat_data["median"].tolist(),
        name="Median",
        marker_color="#34C48A",
    )
    fig.update_layout(
        title_text=f"{feat_name} — mean and median per cluster",
        xaxis_title="Cluster",
        yaxis_title=feat_name,
        barmode="group",
        margin=dict(l=20, r=20, t=50, b=30),
        height=280,
        legend=dict(orientation="h", y=1.12),
    )
    return fig
