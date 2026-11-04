"""PCA and chart utilities for the Country Disease-Burden Clustering dashboard.

PCA Preprocessing Note
-----------------------
All PCA functions that receive ``X`` expect the **standardised** feature matrix
produced by ``src.data.prepare_features``.  Standardisation (zero mean, unit
variance via ``StandardScaler``) is applied before clustering and before PCA so
that indicators measured on very different scales (e.g. deaths per 100 k vs.
DALYs per 100 k) contribute proportionally to principal components.

Interpreting PCA Coefficients (Loadings)
-----------------------------------------
``pca.components_`` stores the eigenvectors of the covariance matrix of the
standardised data.  Each row is one principal component; each column is one
feature.  The value ``components_[i, j]`` is the **linear coefficient** (weight)
of feature *j* in principal component *i*.

* A **large absolute coefficient** means the feature contributes strongly to
  that component's direction in feature space.
* The **sign** indicates direction within the component, but the overall sign
  of a PCA component is arbitrary and can be flipped without changing the PCA
  solution.
* These coefficients are *not* correlation coefficients between the original
  feature and the principal component scores, although they are proportional to
  those correlations when the data are standardised.
* A high loading is **not** evidence of causality or a measure of disease burden.

Scree Plot Interpretation
--------------------------
A scree plot shows how much variance each principal component captures.  It
helps decide how many components are needed to summarise the data.  A "bend"
or "elbow" in the curve is often used as a heuristic, but there is no universal
rule.  The two components used for the scatter plot are chosen for
visualisability, not because they necessarily capture sufficient variance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA


def _apply_dashboard_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,15,19,0.38)",
        font=dict(color="#F2F0EB", family="Inter, sans-serif"),
        title_font=dict(color="#F2F0EB"),
        legend=dict(font=dict(color="#D7D8DA")),
    )
    fig.update_xaxes(gridcolor="rgba(165,169,176,0.15)", zerolinecolor="rgba(165,169,176,0.25)")
    fig.update_yaxes(gridcolor="rgba(165,169,176,0.15)", zerolinecolor="rgba(165,169,176,0.25)")
    return fig


# ── PCA fitting helper ────────────────────────────────────────────────────────

def fit_pca(X: np.ndarray, n_components: int | None = None) -> tuple[PCA, np.ndarray]:
    """Fit a PCA model on the standardised feature matrix and return model + scores.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        **Standardised** feature matrix (output of ``prepare_features``).
    n_components : int or None
        Number of components to retain.  ``None`` means ``min(n_samples,
        n_features)``, which gives the maximum number of useful components
        and is required for a complete scree plot.  Pass ``2`` when only
        the two-dimensional scatter is needed.

    Returns
    -------
    pca : sklearn.decomposition.PCA
        Fitted PCA model.  Access ``pca.explained_variance_ratio_``,
        ``pca.components_``, etc.
    coords : ndarray of shape (n_samples, n_components)
        Projected scores (the PCA coordinates of each observation).

    Raises
    ------
    ValueError
        If ``X`` has fewer than 2 samples or fewer than 2 features.
    """
    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError(
            f"PCA requires at least 2 samples; got {n_samples}."
        )
    if n_features < 2:
        raise ValueError(
            f"PCA requires at least 2 features; got {n_features}."
        )
    # The maximum number of non-trivial components is min(n_samples, n_features).
    max_components = min(n_samples, n_features)
    if n_components is None:
        n_components = max_components
    else:
        n_components = max(1, min(int(n_components), max_components))

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    return pca, coords


# ── PCA scatter plot ──────────────────────────────────────────────────────────

def make_pca_figure(
    X: np.ndarray,
    labels: np.ndarray,
    country_names: list[str],
    pca_model: PCA | None = None,
) -> go.Figure:
    """Return a 2-D PCA scatter plot coloured by cluster assignment.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Scaled feature matrix (must match ``country_names`` length).
    labels : ndarray of shape (n_samples,)
        Integer cluster assignment for each observation.
    country_names : list[str]
        Ordered list of country identifiers aligned with ``X`` rows.
    pca_model : PCA or None
        A pre-fitted scikit-learn PCA model.  If provided, ``X`` is
        transformed with this model (no refitting).  If ``None``, a fresh
        ``PCA(n_components=2)`` is fitted on ``X``.  Pass the model returned
        by :func:`fit_pca` so that the scatter axes are consistent with the
        scree plot and loadings chart.

    Raises
    ------
    ValueError
        If ``len(country_names) != len(labels)``.
    """
    if len(country_names) != len(labels):
        raise ValueError(
            f"Country names count ({len(country_names)}) does not match "
            f"cluster labels count ({len(labels)})."
        )

    if pca_model is not None:
        # Use the pre-fitted model; only transform (no refit)
        coords = pca_model.transform(X)[:, :2]
        var_ratio = pca_model.explained_variance_ratio_[:2]
    else:
        pca = PCA(n_components=min(2, X.shape[1]), random_state=42)
        coords = pca.fit_transform(X)
        var_ratio = pca.explained_variance_ratio_

    explained = var_ratio.sum() * 100
    pc1_label = f"PC1 ({var_ratio[0]*100:.1f}%)"
    pc2_label = f"PC2 ({var_ratio[1]*100:.1f}%)" if len(var_ratio) > 1 else "PC2"

    df = pd.DataFrame({
        pc1_label: coords[:, 0],
        pc2_label: coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(coords)),
        "cluster": labels.astype(str),
        "country": country_names,
    })

    fig = px.scatter(
        df, x=pc1_label, y=pc2_label, color="cluster", hover_name="country",
        title=(
            f"PCA projection — first two components explain {explained:.1f}% of variance"
            "<br><sup>Clustering was performed on all selected features, not on these two axes.</sup>"
        ),
        labels={"cluster": "Cluster"},
        color_discrete_sequence=["#D6AA58", "#B58A3B", "#78A88B", "#A5A9B0", "#F2F0EB"],
    )
    fig.update_layout(legend_title_text="Cluster", margin=dict(l=20, r=20, t=70, b=20))
    return _apply_dashboard_theme(fig)


# ── Scree plot ────────────────────────────────────────────────────────────────

def make_scree_figure(pca_model: PCA) -> go.Figure:
    """Return a dual-panel Plotly figure: scree plot + cumulative explained variance.

    Both panels share the same x-axis (principal component index).

    Parameters
    ----------
    pca_model : PCA
        A fitted scikit-learn PCA model.  The ``explained_variance_ratio_``
        attribute is used directly; no data is needed.

    Returns
    -------
    plotly.graph_objects.Figure

    Notes
    -----
    * The scree plot shows how much variance each component individually
      captures.  It helps assess whether a small number of components
      summarises most of the variation.
    * The cumulative curve shows the total variance retained if you keep the
      first *k* components.
    * A "bend" (elbow) in the scree plot is a common heuristic for choosing
      the number of components, but there is no universally correct threshold.
    * Neither plot automatically determines the right number of clusters.
    """
    var_ratio = pca_model.explained_variance_ratio_
    n_comp = len(var_ratio)
    comp_labels = [f"PC{i+1}" for i in range(n_comp)]
    cumulative = np.cumsum(var_ratio) * 100
    individual = var_ratio * 100

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Individual explained variance (%) per component",
            "Cumulative explained variance (%)",
        ),
        shared_xaxes=False,
    )

    # Left panel — bar chart of individual variance
    fig.add_trace(
        go.Bar(
            x=comp_labels,
            y=individual.tolist(),
            name="Variance %",
            marker_color="#4F8EF7",
            text=[f"{v:.1f}%" for v in individual],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Explains: %{y:.2f}%<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.update_xaxes(title_text="Principal component", row=1, col=1)
    fig.update_yaxes(title_text="Explained variance (%)", range=[0, max(individual) * 1.15], row=1, col=1)

    # Right panel — cumulative line
    fig.add_trace(
        go.Scatter(
            x=comp_labels,
            y=cumulative.tolist(),
            mode="lines+markers",
            name="Cumulative %",
            line=dict(color="#34C48A", width=2),
            marker=dict(size=8),
            text=[f"{v:.1f}%" for v in cumulative],
            hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.2f}%<extra></extra>",
        ),
        row=1, col=2,
    )
    # 90 % and 95 % reference lines
    for ref_pct, dash in [(90, "dot"), (95, "dash")]:
        fig.add_hline(
            y=ref_pct,
            line=dict(color="rgba(200,80,80,0.6)", dash=dash, width=1),
            row=1, col=2,
            annotation_text=f"{ref_pct}%",
            annotation_position="top right",
            annotation_font_size=10,
        )
    fig.update_xaxes(title_text="Components retained", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative explained variance (%)", range=[0, 105], row=1, col=2)

    fig.update_layout(
        title_text="PCA Scree Plot — explained variance by principal component",
        showlegend=False,
        margin=dict(l=40, r=30, t=90, b=50),
        height=370,
    )
    return _apply_dashboard_theme(fig)


# ── Evaluation curves ─────────────────────────────────────────────────────────

def make_evaluation_curves(
    metrics_df: pd.DataFrame,
    algorithm: str = "K-Means (Euclidean)",
    selected_k: int | None = None,
) -> go.Figure:
    """Return a three-panel Plotly figure with Elbow, Silhouette, and Davies-Bouldin curves.

    Panels:
    * Left  — WCSS / L1-objective vs k (Elbow curve).
    * Centre — Silhouette score vs k (higher is better).
    * Right  — Davies-Bouldin index vs k (lower is better, K-Means only).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Output of ``evaluate_kmeans`` or ``evaluate_kmedians`` ``["metrics"]`` key.
        Must contain columns ``k`` and at least one of ``wcss_inertia``,
        ``kmedians_l1_objective``, ``silhouette_score``, or ``davies_bouldin_index``.
    algorithm : str
        Human-readable algorithm label used in axis and panel titles.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    k = metrics_df["k"].tolist()
    is_kmedians = "kmedians_l1_objective" in metrics_df.columns

    has_elbow = True  # always present
    has_silhouette = "silhouette_score" in metrics_df.columns
    has_db = "davies_bouldin_index" in metrics_df.columns

    n_panels = sum([has_elbow, has_silhouette, has_db])
    panel_titles = []
    if has_elbow:
        objective_label = "L1 Objective (Manhattan)" if is_kmedians else "WCSS (Inertia)"
        panel_titles.append(f"Elbow Curve — {objective_label}")
    if has_silhouette:
        panel_titles.append("Silhouette Score (↑ better)")
    if has_db:
        panel_titles.append("Davies–Bouldin Index (↓ better)")

    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=panel_titles)

    col = 1
    marker_kwargs = dict(mode="lines+markers", marker=dict(size=8))

    if has_elbow:
        y_elbow = (
            metrics_df["kmedians_l1_objective"].tolist()
            if is_kmedians
            else metrics_df["wcss_inertia"].tolist()
        )
        fig.add_trace(
            go.Scatter(x=k, y=y_elbow, name="Objective", line=dict(color="#D6AA58"), **marker_kwargs),
            row=1, col=col,
        )
        fig.update_xaxes(title_text="k (clusters)", row=1, col=col)
        fig.update_yaxes(title_text="L1 Objective" if is_kmedians else "WCSS", row=1, col=col)
        col += 1

    if has_silhouette:
        y_sil = metrics_df["silhouette_score"].tolist()
        fig.add_trace(
            go.Scatter(x=k, y=y_sil, name="Silhouette", line=dict(color="#78A88B"), **marker_kwargs),
            row=1, col=col,
        )
        fig.update_xaxes(title_text="k (clusters)", row=1, col=col)
        fig.update_yaxes(title_text="Silhouette Score", row=1, col=col)
        col += 1

    if has_db:
        y_db = metrics_df["davies_bouldin_index"].tolist()
        fig.add_trace(
            go.Scatter(x=k, y=y_db, name="Davies-Bouldin", line=dict(color="#B58A3B"), **marker_kwargs),
            row=1, col=col,
        )
        fig.update_xaxes(title_text="k (clusters)", row=1, col=col)
        fig.update_yaxes(title_text="DB Index", row=1, col=col)

    if selected_k is not None:
        for panel in range(1, n_panels + 1):
            fig.add_vline(
                x=selected_k,
                line=dict(color="#F5A623", dash="dash", width=1),
                row=1,
                col=panel,
            )

    fig.update_layout(
        title_text=f"{algorithm} — Evaluation Curves",
        showlegend=False,
        margin=dict(l=40, r=20, t=80, b=40),
        height=350,
    )
    return _apply_dashboard_theme(fig)


def make_dashboard_evaluation_figure(
    metrics_df: pd.DataFrame,
    algorithm: str,
    selected_k: int,
) -> go.Figure:
    """Show objective and validation metrics on separate, readable scales."""
    is_kmedians = "kmedians_l1_objective" in metrics_df.columns
    objective_column = "kmedians_l1_objective" if is_kmedians else "wcss_inertia"
    objective_label = "L1 objective" if is_kmedians else "WCSS (inertia)"
    has_db = "davies_bouldin_index" in metrics_df.columns
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        specs=[[{}], [{"secondary_y": has_db}]],
        row_heights=[0.42, 0.58],
        vertical_spacing=0.2,
        subplot_titles=(objective_label, "Validation metrics"),
    )
    k_values = metrics_df["k"].tolist()
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=metrics_df[objective_column],
            name=objective_label,
            mode="lines+markers",
            line=dict(color="#F5A623", width=2),
            hovertemplate="k=%{x}<br>" + objective_label + ": %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=metrics_df["silhouette_score"],
            name="Silhouette (higher is better)",
            mode="lines+markers",
            line=dict(color="#29D391", width=2),
            hovertemplate="k=%{x}<br>Silhouette: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    if has_db:
        fig.add_trace(
            go.Scatter(
                x=k_values,
                y=metrics_df["davies_bouldin_index"],
                name="Davies–Bouldin (lower is better)",
                mode="lines+markers",
                line=dict(color="#2EA8FF", width=2),
                hovertemplate="k=%{x}<br>Davies–Bouldin: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
    fig.add_vline(
        x=selected_k,
        line=dict(color="#FFC45C", dash="dash", width=1),
        row=1,
        col=1,
    )
    fig.add_vline(
        x=selected_k,
        line=dict(color="#FFC45C", dash="dash", width=1),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Number of clusters (K)", row=2, col=1)
    fig.update_yaxes(title_text=objective_label, row=1, col=1)
    fig.update_yaxes(title_text="Silhouette", row=2, col=1, secondary_y=False)
    if has_db:
        fig.update_yaxes(title_text="Davies–Bouldin", row=2, col=1, secondary_y=True)
    fig.update_layout(
        title=f"{algorithm} — evaluation by cluster count",
        height=410,
        margin=dict(l=50, r=50, t=70, b=95),
        legend=dict(orientation="h", yanchor="top", y=-0.24, x=0),
    )
    return _apply_dashboard_theme(fig)


# ── PCA loadings ──────────────────────────────────────────────────────────────

def make_pca_loadings_figure(
    X: np.ndarray,
    feature_names: list[str],
    pca_model: PCA | None = None,
    max_components_heatmap: int = 10,
) -> go.Figure:
    """Return a PCA loadings visualisation.

    For two or fewer components a **grouped horizontal bar chart** is used.
    For three or more components a **heatmap** (features × components) is
    preferred and is returned.  If the number of computed components exceeds
    ``max_components_heatmap``, only the first ``max_components_heatmap``
    columns are shown in the heatmap while the complete explained-variance
    summary remains in :func:`make_scree_figure`.

    What the loadings represent
    ---------------------------
    The plotted values are **PCA component coefficients** — the elements of
    ``pca.components_`` (shape: n_components × n_features).  They are the
    linear weights that define each principal component as a weighted sum of
    the standardised features.  Larger absolute values indicate greater
    contribution to the component direction.  Signs are internally consistent
    within a component but the overall sign of each component is arbitrary.
    These are *not* Pearson correlations, though they are proportional to
    correlations when the data are standardised.

    Parameters
    ----------
    X : ndarray
        Standardised feature matrix.  Only used if ``pca_model`` is ``None``.
    feature_names : list[str]
        Names of the features corresponding to columns of ``X``.
    pca_model : PCA or None
        Pre-fitted PCA model.  If provided, ``X`` is ignored (no refit).
        Pass the model from :func:`fit_pca` for consistency across charts.
    max_components_heatmap : int
        Maximum number of components to display in the heatmap.  Default 10.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if pca_model is not None:
        loadings = pca_model.components_
        var_ratio = pca_model.explained_variance_ratio_
    else:
        _pca = PCA(n_components=min(2, X.shape[1]), random_state=42)
        _pca.fit(X)
        loadings = _pca.components_
        var_ratio = _pca.explained_variance_ratio_

    n_comp = loadings.shape[0]
    comp_labels = [f"PC{i+1} ({var_ratio[i]*100:.1f}%)" for i in range(n_comp)]

    if n_comp <= 2:
        # Grouped horizontal bar chart for PC1 / PC2
        df_rows = []
        for comp_idx in range(n_comp):
            for feat_idx, feat in enumerate(feature_names):
                df_rows.append({
                    "Component": comp_labels[comp_idx],
                    "Feature": feat,
                    "Loading coefficient": loadings[comp_idx, feat_idx],
                })
        df_load = pd.DataFrame(df_rows)
        fig = px.bar(
            df_load,
            x="Loading coefficient",
            y="Feature",
            color="Component",
            barmode="group",
            orientation="h",
            title="PCA Component Coefficients (PC1 and PC2)",
            color_discrete_sequence=["#D6AA58", "#78A88B"],
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=60, b=20),
            height=max(300, 30 * len(feature_names) + 80),
        )
        return _apply_dashboard_theme(fig)

    # Heatmap for three or more components
    display_comps = min(n_comp, max_components_heatmap)
    z = loadings[:display_comps].T  # shape: (n_features, display_comps)
    x_labels = comp_labels[:display_comps]

    # Build hover text matrix
    hover = []
    for feat_idx, feat in enumerate(feature_names):
        row_hover = []
        for comp_idx in range(display_comps):
            val = loadings[comp_idx, feat_idx]
            row_hover.append(
                f"Feature: {feat}<br>Component: PC{comp_idx+1}<br>Coefficient: {val:.4f}"
            )
        hover.append(row_hover)

    fig = go.Figure(
        go.Heatmap(
            z=z.tolist(),
            x=x_labels,
            y=feature_names,
            colorscale="RdBu",
            zmid=0,
            text=hover,
            hoverinfo="text",
            colorbar=dict(
                title="Loading<br>coefficient",
                tickformat=".2f",
            ),
        )
    )
    title = "PCA Component Coefficients"
    if display_comps < n_comp:
        title += f" (showing first {display_comps} of {n_comp} components)"
    fig.update_layout(
        title_text=title,
        xaxis_title="Principal component",
        yaxis_title="Feature",
        margin=dict(l=20, r=20, t=70, b=40),
        height=max(350, 30 * len(feature_names) + 120),
    )
    return _apply_dashboard_theme(fig)


def make_cluster_profile_figure(
    profile_df: pd.DataFrame,
    feature_name: str,
) -> go.Figure:
    """Compare original-scale feature means and medians across clusters."""
    feature_data = profile_df.loc[profile_df["feature"] == feature_name].sort_values("cluster")
    if feature_data.empty:
        raise ValueError(f"No cluster profile is available for feature '{feature_name}'.")

    fig = go.Figure()
    fig.add_bar(
        x=feature_data["cluster"].astype(str),
        y=feature_data["mean"],
        name="Mean",
        marker_color="#D6AA58",
        error_y=dict(
            type="data",
            array=feature_data["std"].fillna(0).tolist(),
            visible=True,
        ),
    )
    fig.add_scatter(
        x=feature_data["cluster"].astype(str),
        y=feature_data["median"],
        name="Median",
        mode="markers",
        marker=dict(color="#78A88B", size=11, symbol="diamond"),
    )
    fig.update_layout(
        title=f"{feature_name} by cluster (original values)",
        xaxis_title="Cluster",
        yaxis_title=feature_name,
        margin=dict(l=20, r=20, t=60, b=30),
        height=360,
    )
    return _apply_dashboard_theme(fig)


def make_country_cluster_map(
    country_names: list[str],
    labels: np.ndarray,
) -> go.Figure:
    """Show assigned countries on a world map with a discrete cluster legend."""
    if len(country_names) != len(labels):
        raise ValueError(
            f"Country names count ({len(country_names)}) does not match "
            f"cluster labels count ({len(labels)})."
        )

    cluster_colors = ["#F5A623", "#29D391", "#2EA8FF", "#C56BE0", "#E16B60"]
    cluster_values = sorted(np.unique(labels).tolist())
    fig = go.Figure()
    for cluster_index, cluster in enumerate(cluster_values):
        locations = [
            country for country, assigned_cluster in zip(country_names, labels)
            if assigned_cluster == cluster
        ]
        color = cluster_colors[cluster_index % len(cluster_colors)]
        fig.add_trace(
            go.Choropleth(
                locations=locations,
                locationmode="country names",
                z=[0] * len(locations),
                zmin=0,
                zmax=1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=f"Cluster {int(cluster) + 1}",
                customdata=[[f"Cluster {int(cluster) + 1}"]] * len(locations),
                hovertemplate="%{location}<br>%{customdata[0]}<extra></extra>",
                marker_line_color="rgba(10, 13, 16, 0.8)",
                marker_line_width=0.35,
            )
        )

    fig.update_geos(
        showframe=False,
        showcoastlines=False,
        showcountries=True,
        countrycolor="rgba(190, 160, 110, 0.18)",
        showland=True,
        landcolor="rgba(29, 34, 37, 0.9)",
        showocean=True,
        oceancolor="rgba(8, 11, 13, 0.55)",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title="Country clusters — global view",
        height=330,
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(title_text="Assigned cluster"),
    )
    return _apply_dashboard_theme(fig)


def make_standardized_cluster_profiles(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> go.Figure:
    """Compare cluster means in standardized feature space without refitting."""
    if len(X) != len(labels):
        raise ValueError(
            f"Feature rows count ({len(X)}) does not match cluster labels count ({len(labels)})."
        )
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError("Feature names must correspond to the columns of the feature matrix.")

    profile_data = pd.DataFrame(X, columns=feature_names)
    profile_data["cluster"] = labels
    means = profile_data.groupby("cluster", sort=True)[feature_names].mean()
    cluster_names = [f"Cluster {int(cluster) + 1}" for cluster in means.index]
    fig = go.Figure(
        go.Heatmap(
            z=means.to_numpy(),
            x=feature_names,
            y=cluster_names,
            colorscale="RdYlBu_r",
            zmid=0,
            colorbar=dict(title="Mean z-score"),
            hovertemplate=(
                "%{y}<br>%{x}<br>Standardized mean: %{z:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Standardized disease-burden profiles",
        xaxis_title="Selected indicator",
        yaxis_title="Assigned cluster",
        height=max(300, 58 * len(cluster_names) + 120),
        margin=dict(l=18, r=12, t=48, b=75),
    )
    fig.update_xaxes(tickangle=-35)
    return _apply_dashboard_theme(fig)
