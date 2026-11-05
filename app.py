from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

from src.clustering import compute_cluster_profiles, evaluate_kmeans, evaluate_kmedians
from src.data import prepare_features, validate_dataset
from src.gbd_transform import transform_gbd_long_to_wide
from src.visualization import (
    fit_pca,
    make_cluster_profile_figure,
    make_country_cluster_map,
    make_dashboard_evaluation_figure,
    make_evaluation_curves,
    make_pca_figure,
    make_pca_loadings_figure,
    make_scree_figure,
    make_standardized_cluster_profiles,
)


st.set_page_config(
    page_title="Country Disease-Burden Clustering",
    page_icon="🌍",
    layout="wide",
)

PROJECT_DIR = Path(__file__).parent
LOCAL_GBD_PATH = PROJECT_DIR / "data" / "raw" / "IHME-GBD_2023_DATA-34567faa-1.csv"


def _clear_clustering_result() -> None:
    st.session_state.pop("clustering_result", None)
    st.session_state.pop("selected_k_solution", None)


def _forget_dataset() -> None:
    _clear_clustering_result()
    for key in (
        "source_df",
        "source_country_col",
        "source_features",
        "source_metadata",
        "source_signature",
        "source_content_hash",
    ):
        st.session_state.pop(key, None)
        
        
css_path = PROJECT_DIR / "assets" / "style.css"

if css_path.is_file():
    background_image_url = (
        "https://raw.githubusercontent.com/"
        "Hi-manshu03/country-disease-burden-clustering/"
        "main/static/dashboard-bg.png"
    )

    css = css_path.read_text(encoding="utf-8")
    css = css.replace("dashboard-bg.png", background_image_url)

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
else:
    st.warning(
        "Custom stylesheet not found; "
        "the dashboard is using Streamlit's dark theme."
    )

navigation_sections = [
        "Overview",
        "Dataset & Ingestion",
        "Algorithm & Configuration",
        "Model Evaluation",
        "Dimensionality Reduction",
        "Cluster Profiles",
        "Reports & Export",
]
navigation_icons = {
    "Overview": "▦",
    "Dataset & Ingestion": "▤",
    "Algorithm & Configuration": "☷",
    "Model Evaluation": "▥",
    "Dimensionality Reduction": "⌁",
    "Cluster Profiles": "◈",
    "Reports & Export": "⇩",
}
with st.sidebar:
        st.markdown(
                '<div class="sidebar-brand"><span class="brand-mark">◎</span>'
                '<span>PUBLIC HEALTH<br>INTELLIGENCE</span></div>',
                unsafe_allow_html=True,
        )
        st.markdown(
                '<div class="sidebar-product"><span class="product-mark">⌁</span>'
                '<strong>Country Disease-Burden Pattern Clustering</strong></div>',
                unsafe_allow_html=True,
        )
        selected_section = st.radio(
                "Dashboard sections",
                navigation_sections,
                key="navigation_choice",
                format_func=lambda label: f"{navigation_icons[label]}  {label}",
                label_visibility="collapsed",
        )
        st.session_state["dashboard_tabs"] = selected_section

st.markdown(
        """
        <header class="project-header">
            <div class="project-eyebrow">PUBLIC HEALTH ANALYTICS</div>
            <h1 class="project-title">Country Disease-Burden Pattern Clustering</h1>
            <p class="project-description">
                Identify groups of countries with similar disease-burden profiles using
                unsupervised machine learning.
            </p>
            <div class="project-meta">IHME GBD 2023 &nbsp; | &nbsp; K-MEANS &nbsp; | &nbsp; K-MEDIANS &nbsp; | &nbsp; PCA</div>
        </header>
        """,
        unsafe_allow_html=True,
)
st.warning(
    "Clusters describe statistical similarities only. They do not establish causes, "
    "rank health systems, or provide diagnoses."
)

(
    overview_tab,
    dataset_tab,
    configuration_tab,
    evaluation_tab,
    pca_tab,
    profiles_tab,
    exports_tab,
) = st.tabs(
    navigation_sections,
    key="dashboard_tabs",
    on_change="rerun",
)

with dataset_tab:
    st.subheader("Dataset & ingestion")
    st.caption(
        "Upload a prepared country-by-feature CSV or an IHME GBD long-format export. "
        "GBD exports are filtered to DALY rates, both sexes, age-standardized estimates, "
        "and one selected year before pivoting."
    )

    source_options = ["Upload a CSV"]
    if LOCAL_GBD_PATH.is_file():
        source_options.append("Use supplied local GBD 2023 extract")
    data_source = st.radio(
        "Data source",
        source_options,
        horizontal=True,
        key="data_source_choice",
    )

    file_bytes: bytes | None = None
    source_name = ""
    if data_source == "Use supplied local GBD 2023 extract":
        try:
            file_bytes = LOCAL_GBD_PATH.read_bytes()
            source_name = LOCAL_GBD_PATH.name
        except OSError as exc:
            st.error(f"Could not read the local GBD file: {exc}")
            _forget_dataset()
    else:
        uploaded = st.file_uploader(
            "Choose a country-level or GBD long-format CSV",
            type=["csv"],
            key="dataset_upload",
            help="CSV files up to 25 MB are supported.",
        )
        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            source_name = uploaded.name

    if file_bytes is None:
        if "source_df" not in st.session_state:
            st.info(
                "No dataset is loaded yet. The local GBD option is available only when its "
                "authorized source file is present beside the application."
            )
        else:
            st.success(
                f"Loaded {len(st.session_state['source_df']):,} country observations from "
                f"{st.session_state['source_metadata']['label']}."
            )
    else:
        content_hash = sha256(file_bytes).hexdigest()
        if st.session_state.get("source_content_hash") not in (None, content_hash):
            _forget_dataset()
        st.session_state["source_content_hash"] = content_hash

        try:
            incoming_df = pd.read_csv(BytesIO(file_bytes))
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            _forget_dataset()
            st.error(f"Could not parse the CSV: {exc}")
            incoming_df = None

        if incoming_df is not None:
            is_gbd_long = {"location_name", "cause_name", "val"}.issubset(incoming_df.columns)
            target_year: int | None = None
            metadata = {
                "label": "Prepared country-level CSV",
                "source_name": source_name,
                "kind": "prepared",
            }

            if is_gbd_long:
                available_years: list[int] = []
                if "year" in incoming_df.columns:
                    year_values = pd.to_numeric(incoming_df["year"], errors="coerce").dropna()
                    available_years = sorted({int(year) for year in year_values})
                if available_years:
                    target_year = st.selectbox(
                        "GBD estimate year",
                        available_years,
                        index=len(available_years) - 1,
                        key=f"gbd_year_{content_hash[:12]}",
                    )
                try:
                    prepared_wide = transform_gbd_long_to_wide(
                        incoming_df,
                        target_year=target_year,
                    )
                except ValueError as exc:
                    if st.session_state.get("source_signature") != f"{content_hash}:{target_year}":
                        _forget_dataset()
                    st.error(f"GBD transformation failed: {exc}")
                    prepared_wide = None
                metadata = {
                    "label": "IHME GBD long-format CSV",
                    "source_name": source_name,
                    "kind": "gbd",
                    "year": target_year,
                    "measure": "DALYs",
                    "metric": "Rate",
                    "age": "Age-standardized",
                    "sex": "Both",
                }
            else:
                prepared_wide = incoming_df

            if prepared_wide is not None:
                try:
                    country_col, numeric_cols = validate_dataset(prepared_wide)
                except ValueError as exc:
                    if st.session_state.get("source_content_hash") == content_hash:
                        _forget_dataset()
                    st.error(f"Dataset validation failed: {exc}")
                else:
                    prepared_wide = prepared_wide.reset_index(drop=True)
                    signature = f"{content_hash}:{target_year if is_gbd_long else 'prepared'}"
                    if signature != st.session_state.get("source_signature"):
                        _clear_clustering_result()
                    st.session_state["source_df"] = prepared_wide
                    st.session_state["source_country_col"] = country_col
                    st.session_state["source_features"] = numeric_cols
                    st.session_state["source_metadata"] = metadata
                    st.session_state["source_signature"] = signature
                    st.session_state["source_content_hash"] = content_hash
                    st.success(
                        f"Validated {len(prepared_wide):,} country observations and "
                        f"{len(numeric_cols):,} numeric indicators."
                    )

    if "source_df" in st.session_state:
        source_df = st.session_state["source_df"]
        metadata = st.session_state["source_metadata"]
        st.markdown("#### Dataset summary")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Country observations", f"{len(source_df):,}")
        summary_cols[1].metric("Numeric indicators", f"{len(st.session_state['source_features']):,}")
        summary_cols[2].metric("Source format", metadata["kind"].upper())
        summary_cols[3].metric("Selected year", str(metadata.get("year", "Not specified")))
        if metadata["kind"] == "gbd":
            st.caption(
                f"Filtered dimensions: {metadata['measure']} · {metadata['metric']} · "
                f"{metadata['age']} · {metadata['sex']}. Estimates are rates per 100,000 "
                "in the GBD export. Confirm source metadata before interpreting."
            )
        else:
            st.caption(
                "Confirm that all selected indicators use comparable years, measures, "
                "age groups, sexes and units before analysis."
            )
        with st.expander("Preview country-level data"):
            st.dataframe(source_df.head(20), width="stretch")


source_df = st.session_state.get("source_df")
prepared = None
analysis_signature = None
results = st.session_state.get("clustering_result")
selected_k: int | None = None
features: list[str] = []
profile_df = None
assignments = None

with configuration_tab:
    st.subheader("Algorithm & configuration")
    if source_df is None:
        st.info("Load and validate a dataset in **Dataset & Ingestion** before configuring an experiment.")
    else:
        country_col = st.session_state["source_country_col"]
        numeric_cols = st.session_state["source_features"]
        source_signature = st.session_state["source_signature"]
        st.caption("Features are standardized after row-level missingness filtering and median imputation.")
        features = st.multiselect(
            "Disease-burden features",
            numeric_cols,
            default=numeric_cols[: min(5, len(numeric_cols))],
            key="selected_features",
        )
        k_values = st.multiselect(
            "Cluster counts (k)",
            options=list(range(2, 9)),
            default=[2, 3, 4],
            key="selected_k_values",
        )
        algorithm_choice = st.selectbox(
            "Primary clustering algorithm",
            options=["K-Means (Euclidean)", "K-Medians (Manhattan)"],
            help=(
                "K-Means minimizes squared Euclidean distances with means as centroids. "
                "K-Medians minimizes Manhattan distances with feature-wise medians."
            ),
            key="algorithm_choice",
        )
        compare_algorithms = st.checkbox(
            "Evaluate both algorithms on the same features and k values",
            value=False,
            key="compare_algorithms",
            help="The objective values use different distance definitions and are not directly comparable.",
        )
        missing_threshold_pct = st.slider(
            "Maximum missing features allowed per country (%)",
            min_value=0,
            max_value=100,
            value=30,
            step=5,
            key="missing_threshold_pct",
            help="Rows above this threshold are excluded; remaining missing cells use median imputation.",
        )
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
            key="random_state",
        )
        max_missing_fraction = missing_threshold_pct / 100.0
        algorithms_to_run = (
            ["K-Means (Euclidean)", "K-Medians (Manhattan)"]
            if compare_algorithms
            else [algorithm_choice]
        )
        analysis_signature = (
            source_signature,
            tuple(features),
            tuple(k_values),
            tuple(algorithms_to_run),
            float(max_missing_fraction),
            int(random_state),
        )
        current_result = st.session_state.get("clustering_result")
        if current_result is not None and current_result.get("analysis_signature") != analysis_signature:
            _clear_clustering_result()

        if len(features) < 2:
            st.info("Select at least two numeric features.")
        elif len(k_values) < 3:
            st.info("Select at least three cluster counts; counts greater than the usable sample size are omitted.")
        else:
            try:
                prepared = prepare_features(
                    source_df,
                    list(features),
                    max_missing_fraction=max_missing_fraction,
                    country_col=country_col,
                )
            except ValueError as exc:
                st.error(f"Preprocessing failed: {exc}")
            else:
                summary_cols = st.columns(4)
                summary_cols[0].metric("Input countries", f"{len(source_df):,}")
                summary_cols[1].metric("Retained", f"{prepared['retained_rows']:,}")
                summary_cols[2].metric("Excluded", f"{prepared['excluded_rows']:,}")
                summary_cols[3].metric("Imputed cells", f"{prepared['imputed_cells']:,}")
                if prepared["excluded_rows"]:
                    st.warning(
                        f"{prepared['excluded_rows']} country observation(s) exceeded the "
                        f"{missing_threshold_pct}% missingness threshold. Exclusion can change "
                        "the geographic and population balance represented."
                    )
                    st.dataframe(prepared["exclusions_df"], width="stretch")
                    st.download_button(
                        "Download exclusion audit (CSV)",
                        data=prepared["exclusions_df"].to_csv(index=False).encode("utf-8"),
                        file_name="excluded_countries_audit.csv",
                        mime="text/csv",
                        key="dl_exclusions",
                    )
                if st.button("Run clustering experiments", type="primary"):
                    analysis_raw = source_df.iloc[prepared["row_indices"]].reset_index(drop=True)
                    try:
                        pca_model, _ = fit_pca(prepared["X"])
                        algorithm_results = {}
                        with st.spinner("Fitting clustering models and calculating metrics..."):
                            for algorithm in algorithms_to_run:
                                if algorithm == "K-Means (Euclidean)":
                                    run = evaluate_kmeans(
                                        prepared["X"],
                                        list(k_values),
                                        random_state=int(random_state),
                                    )
                                else:
                                    run = evaluate_kmedians(
                                        prepared["X"],
                                        list(k_values),
                                        random_state=int(random_state),
                                    )
                                algorithm_results[algorithm] = run
                    except ValueError as exc:
                        st.error(f"Experiment failed: {exc}")
                    else:
                        main_run = algorithm_results[algorithm_choice]
                        omitted_k = [int(k) for k in k_values if int(k) not in main_run["labels_by_k"]]
                        if omitted_k:
                            st.warning(
                                f"Cluster count(s) {omitted_k} were omitted because k must be "
                                f"strictly less than the {prepared['retained_rows']} retained countries."
                            )
                        if (
                            "selected_k_solution" in st.session_state
                            and st.session_state["selected_k_solution"] not in main_run["labels_by_k"]
                        ):
                            st.session_state.pop("selected_k_solution", None)
                        st.session_state["clustering_result"] = {
                            "analysis_signature": analysis_signature,
                            "algorithm": main_run["algorithm"],
                            "metrics": main_run["metrics"],
                            "labels_by_k": main_run["labels_by_k"],
                            "comparison_metrics": {
                                name: run["metrics"] for name, run in algorithm_results.items()
                            },
                            "comparison_labels": {
                                name: run["labels_by_k"] for name, run in algorithm_results.items()
                            },
                            "X": prepared["X"],
                            "pca_model": pca_model,
                            "analysis_raw": analysis_raw,
                            "country_col": country_col,
                            "country_names": analysis_raw[country_col].astype(str).tolist(),
                            "features": list(features),
                            "k_values": list(k_values),
                            "max_missing_fraction": max_missing_fraction,
                            "missing_threshold_pct": missing_threshold_pct,
                            "algorithm_choice": algorithm_choice,
                            "random_state": int(random_state),
                            "source_metadata": st.session_state["source_metadata"],
                            "source_signature": source_signature,
                            "input_rows": len(source_df),
                            "retained_rows": prepared["retained_rows"],
                            "excluded_rows": prepared["excluded_rows"],
                            "imputed_cells": prepared["imputed_cells"],
                            "exclusions_df": prepared["exclusions_df"],
                        }
                        st.success("Please check Results in Model Evaluation tab.")

results = st.session_state.get("clustering_result")
if results is not None and results.get("analysis_signature") != analysis_signature:
    results = None

with overview_tab:
    selected_features = list(features or st.session_state.get("selected_features", []))
    selected_k_values = list(st.session_state.get("selected_k_values", []))
    dashboard_k = None
    best_silhouette = None
    best_silhouette_algorithm = None
    if results is not None:
        available_k = list(results["labels_by_k"])
        dashboard_k = st.session_state.get("selected_k_solution")
        if dashboard_k not in available_k:
            dashboard_k = available_k[0]
        silhouette_scores = []
        for algorithm, metric_frame in results["comparison_metrics"].items():
            if "silhouette_score" not in metric_frame:
                continue
            scores = pd.to_numeric(metric_frame["silhouette_score"], errors="coerce")
            for score in scores[np.isfinite(scores)]:
                silhouette_scores.append((float(score), algorithm))
        if silhouette_scores:
            best_silhouette, best_silhouette_algorithm = max(silhouette_scores, key=lambda row: row[0])

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "🌐 Countries analysed",
        f"{results['retained_rows']:,}" if results is not None else "—",
        "Validated for model" if results is not None else "Run an experiment",
        delta_color="off",
    )
    metric_cols[1].metric(
        "▦ Indicators (features)",
        f"{len(results['features'] if results is not None else selected_features):,}",
        "Selected" if selected_features else "Select indicators",
        delta_color="off",
    )
    metric_cols[2].metric(
        "◎ Clusters (K)",
        str(dashboard_k if dashboard_k is not None else (selected_k_values[0] if selected_k_values else "—")),
        results["algorithm"] if results is not None else "Not executed",
        delta_color="off",
    )
    metric_cols[3].metric(
        "▥ Best silhouette score",
        f"{best_silhouette:.2f}" if best_silhouette is not None else "—",
        best_silhouette_algorithm or "No valid score available",
        delta_color="off",
    )

    if results is None:
        if source_df is None:
            st.info("Load a validated dataset, configure an experiment, and run clustering to populate this dashboard.")
        else:
            st.info("Configure and run a clustering experiment to display country assignments and model visualizations.")
    else:
        labels = results["labels_by_k"][dashboard_k]
        top_left, top_right = st.columns(2)
        with top_left, st.container(border=True):
            st.markdown("#### Country Clusters — Global View")
            st.plotly_chart(
                make_country_cluster_map(results["country_names"], labels),
                width="stretch",
                key="overview_country_map",
            )
            st.caption("Only countries with geographic boundaries are colored; all assignments remain available in Cluster Profiles and Reports & Export.")
        with top_right, st.container(border=True):
            st.markdown("#### Cluster Profile Comparison")
            st.plotly_chart(
                make_standardized_cluster_profiles(
                    results["X"], labels, results["features"]
                ),
                width="stretch",
                key="overview_cluster_profiles",
            )
            st.caption("Cluster means are shown as standardized values for comparison across indicators.")

        bottom_left, bottom_right = st.columns(2)
        with bottom_left, st.container(border=True):
            st.markdown("#### Disease-Burden Pattern — PCA")
            st.plotly_chart(
                make_pca_figure(
                    results["X"], labels, results["country_names"],
                    pca_model=results["pca_model"],
                ),
                width="stretch",
                key="overview_pca",
            )
        with bottom_right, st.container(border=True):
            st.markdown(f"#### Model Evaluation — Cluster Count (K) · Selected K = {dashboard_k}")
            st.plotly_chart(
                make_dashboard_evaluation_figure(
                    results["metrics"],
                    algorithm=results["algorithm"],
                    selected_k=dashboard_k,
                ),
                width="stretch",
                key="overview_evaluation",
            )

with evaluation_tab:
    st.subheader("Model evaluation")
    if results is None:
        st.info("Configure and run an experiment to view evaluation metrics.")
    else:
        st.caption(
            "WCSS and the K-Medians L1 objective summarize within-cluster distances for their "
            "respective algorithms; their values are not comparable across algorithms. "
            "Silhouette is higher-is-better; Davies–Bouldin is lower-is-better and shown for K-Means."
        )
        st.plotly_chart(
            make_evaluation_curves(
                results["metrics"], algorithm=results["algorithm"], selected_k=selected_k
            ),
            width="stretch",
        )
        st.dataframe(results["metrics"], width="stretch")
        if len(results["comparison_metrics"]) > 1:
            with st.expander("Compare both algorithms"):
                st.warning(
                    "Both algorithms use the same standardized observations and selected k values. "
                    "The squared-Euclidean WCSS and Manhattan L1 objectives have different definitions "
                    "and scales; compare them within an algorithm, not against one another."
                )
                for algorithm, metrics in results["comparison_metrics"].items():
                    st.markdown(f"**{algorithm}**")
                    st.plotly_chart(
                        make_evaluation_curves(metrics, algorithm=algorithm),
                        width="stretch",
                        key=f"comparison_{algorithm}",
                    )
                    st.dataframe(metrics, width="stretch")
        with st.expander("Metric interpretation"):
            st.markdown(
                "- **WCSS / inertia:** sum of squared Euclidean distances to K-Means centroids; "
                "decreases as k increases.\n"
                "- **L1 objective:** sum of Manhattan distances to K-Medians representatives; "
                "it is not WCSS.\n"
                "- **Silhouette:** relative cohesion and separation under the algorithm's "
                "corresponding distance metric; higher values indicate better-separated clusters.\n"
                "- **Davies–Bouldin:** Euclidean cluster dispersion/separation summary for K-Means; "
                "lower values indicate tighter, better-separated clusters."
            )
        available_k = list(results["labels_by_k"])
        if (
            "selected_k_solution" in st.session_state
            and st.session_state["selected_k_solution"] not in available_k
        ):
            st.session_state.pop("selected_k_solution", None)
        selected_k = st.selectbox(
            "Cluster solution to inspect",
            available_k,
            key="selected_k_solution",
        )
        observed_clusters = int(results["metrics"].loc[results["metrics"]["k"] == selected_k, "clusters_observed"].iloc[0])
        if observed_clusters < selected_k:
            st.warning(
                f"The requested k={selected_k} produced {observed_clusters} distinct cluster(s), "
                "usually because several observations are identical."
            )

with pca_tab:
    st.subheader("Dimensionality reduction")
    if results is None:
        st.info("Run an experiment to inspect PCA projections and loadings.")
    else:
        labels = results["labels_by_k"][selected_k]
        st.caption(
            "Countries are projected onto PC1 and PC2 for visualization. Clustering uses all "
            "selected standardized features, not just the two plotted axes."
        )
        st.plotly_chart(
            make_pca_figure(
                results["X"],
                labels,
                results["country_names"],
                pca_model=results["pca_model"],
            ),
            width="stretch",
        )
        scree_fig = make_scree_figure(results["pca_model"])
        st.plotly_chart(scree_fig, width="stretch")
        variance = results["pca_model"].explained_variance_ratio_
        cumulative = variance.cumsum()
        variance_df = pd.DataFrame(
            {
                "Component": [f"PC{i + 1}" for i in range(len(variance))],
                "Explained variance (%)": variance * 100,
                "Cumulative variance (%)": cumulative * 100,
            }
        )
        with st.expander("Explained variance table"):
            st.dataframe(variance_df.round(2), width="stretch")
        st.plotly_chart(
            make_pca_loadings_figure(
                results["X"],
                results["features"],
                pca_model=results["pca_model"],
            ),
            width="stretch",
        )
        st.caption(
            "Loadings are linear coefficients for standardized features, not original disease-burden "
            "values or evidence of causality. A component's sign is arbitrary."
        )

with profiles_tab:
    st.subheader("Cluster profiles")
    if results is None:
        st.info("Run an experiment to inspect country assignments and cluster profiles.")
    else:
        labels = results["labels_by_k"][selected_k]
        profile_df = compute_cluster_profiles(
            raw_df=results["analysis_raw"],
            features=results["features"],
            labels=labels,
        )
        assignments = results["analysis_raw"].copy()
        assignments["cluster"] = labels
        cluster_sizes = (
            assignments.groupby("cluster", as_index=False)
            .agg(country_count=(results["country_col"], "size"))
            .sort_values("cluster")
        )
        st.caption(
            "Feature summaries use original, unscaled values. Clusters are descriptive statistical "
            "groups and should not be interpreted as causal or diagnostic categories."
        )
        st.dataframe(cluster_sizes, width="stretch")
        st.dataframe(profile_df, width="stretch")
        with st.expander("Compare profiles by feature"):
            for feature in results["features"]:
                st.plotly_chart(
                    make_cluster_profile_figure(profile_df, feature),
                    width="stretch",
                    key=f"profile_{feature}",
                )
        cluster_options = sorted(int(cluster) for cluster in assignments["cluster"].unique())
        selected_clusters = st.multiselect(
            "Filter country assignments by cluster",
            cluster_options,
            default=cluster_options,
            key="profile_cluster_filter",
        )
        country_query = st.text_input(
            "Search country or territory",
            key="profile_country_search",
        ).strip().casefold()
        filtered_assignments = assignments.loc[
            assignments["cluster"].isin(selected_clusters)
        ]
        if country_query:
            filtered_assignments = filtered_assignments.loc[
                filtered_assignments[results["country_col"]]
                .astype(str)
                .str.casefold()
                .str.contains(country_query, regex=False)
            ]
        st.dataframe(filtered_assignments, width="stretch")

with exports_tab:
    st.subheader("Reports & export")
    if results is None:
        st.info("Run an experiment to generate exports.")
    else:
        if selected_k is None:
            selected_k = next(iter(results["labels_by_k"]))
        labels = results["labels_by_k"][selected_k]
        export_assignments = results["analysis_raw"].copy()
        export_assignments["cluster"] = labels
        export_profiles = compute_cluster_profiles(
            raw_df=results["analysis_raw"],
            features=results["features"],
            labels=labels,
        )
        st.download_button(
            "Download country assignments (CSV)",
            data=export_assignments.to_csv(index=False).encode("utf-8"),
            file_name=f"country_clusters_k{selected_k}.csv",
            mime="text/csv",
            key="dl_clusters",
        )
        st.download_button(
            "Download cluster profiles (CSV)",
            data=export_profiles.to_csv(index=False).encode("utf-8"),
            file_name=f"cluster_profiles_k{selected_k}.csv",
            mime="text/csv",
            key="dl_profiles",
        )
        st.download_button(
            "Download evaluation metrics (CSV)",
            data=results["metrics"].to_csv(index=False).encode("utf-8"),
            file_name="clustering_evaluation.csv",
            mime="text/csv",
            key="dl_metrics",
        )
        st.download_button(
            "Download exclusion audit (CSV)",
            data=results["exclusions_df"].to_csv(index=False).encode("utf-8"),
            file_name="excluded_countries_audit.csv",
            mime="text/csv",
            key="dl_exclusion_report",
            disabled=results["exclusions_df"].empty,
        )
        report_lines = [
            "Country Disease-Burden Pattern Clustering — Analysis Report",
            "=" * 64,
            f"Source: {results['source_metadata']['source_name']}",
            f"Source type: {results['source_metadata']['label']}",
            f"Selected year: {results['source_metadata'].get('year', 'Not specified')}",
            f"Algorithm: {results['algorithm']}",
            f"Random seed: {results['random_state']}",
            f"Features: {', '.join(results['features'])}",
            f"Selected k: {selected_k}",
            f"Input countries: {results['input_rows']}",
            f"Retained countries: {results['retained_rows']}",
            f"Excluded countries: {results['excluded_rows']}",
            f"Imputed cells: {results['imputed_cells']}",
            "",
            "Evaluation metrics:",
            results["metrics"].to_string(index=False),
            "",
            "Interpretation: Clusters are statistical similarities only. They do not establish "
            "causality, rank health systems, or provide diagnoses.",
        ]
        st.download_button(
            "Download analysis report (TXT)",
            data="\n".join(report_lines).encode("utf-8"),
            file_name="country_clustering_report.txt",
            mime="text/plain",
            key="dl_report",
        )
        st.caption(
            "The report records the selected analysis and observed metrics. Review the data "
            "provenance and limitations before sharing exported results."
        )

with st.sidebar:
    st.markdown('<div class="sidebar-status-title">WORKSPACE STATUS</div>', unsafe_allow_html=True)
    dataset_loaded = source_df is not None
    model_executed = results is not None
    cluster_count = selected_k if selected_k is not None else "—"
    data_source_label = (
        st.session_state["source_metadata"]["label"] if dataset_loaded else "No source"
    )
    st.markdown(
        f'<div class="status-row"><span>Dataset</span><strong class="status-pill {"is-ready" if dataset_loaded else "is-idle"}">'
        f'{"Loaded" if dataset_loaded else "Not loaded"}</strong></div>'
        f'<div class="status-row"><span>Model</span><strong class="status-pill {"is-ready" if model_executed else "is-idle"}">'
        f'{"Executed" if model_executed else "Not run"}</strong></div>'
        f'<div class="status-row"><span>Clusters</span><strong>{cluster_count}</strong></div>'
        f'<div class="status-row"><span>Data source</span><strong class="status-source">{data_source_label}</strong></div>',
        unsafe_allow_html=True,
    )
