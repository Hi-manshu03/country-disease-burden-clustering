# Project 28 — Country Disease-Burden Pattern Clustering for Public Health Analysis

## 1. Project Purpose & Scope
This project provides a reproducible, interactive, and deployable unsupervised-learning dashboard to discover patterns in country-level Global Burden of Disease (GBD) indicators. By clustering countries with comparable epidemiological metrics, public health analysts can explore cross-national burden profiles, identify shared health challenges, and inspect multivariate disease-burden structures.

> [!WARNING]
> **Responsible Epidemiological Interpretation:**  
> Unsupervised clustering identifies statistical similarities and distributional patterns across observations; it **does not** prove causality, etiology, or clinical efficacy. Clusters should not be interpreted as definitive rankings, causal assessments, or clinical diagnoses.

---

## 2. Current Features & Methodological Architecture

### Data Ingestion & Contract Validation
- **Strict country identifier contract:** Requires exactly one observation per country. Automatically recognizes column names such as `country`, `location`, `entity`, `iso3`, `iso_code`, `country_name`, `nation`, `state`, `territory`, or `economy`. Rejects nulls, whitespace, and literal string missing values (`"NaN"`, `"None"`).
- **Duplicate rejection:** Rejects duplicate country identifiers with actionable diagnostic messages. Multi-year, sex-disaggregated, or age-stratified data must be harmonized or filtered before analysis.
- **Robust numeric coercion:** Infinite values (`inf`, `-inf`) and non-numeric strings are safely coerced to missing values (`NaN`).
- **Configurable missingness threshold:** Allows users to adjust the maximum allowed row-level missingness ($0\%$ to $100\%$, defaulting to $30\%$). Rows exceeding the threshold are excluded with an epidemiological caveat regarding sample representativeness.
- **Audit logging:** Excluded countries are logged with their row indices, missing feature counts, missing percentages, and reason codes (`EXCEEDS_MISSING_THRESHOLD` or `ALL_FEATURES_MISSING`), available for inspection and CSV export.
- **Median imputation & standardization:** Remaining missing cells in retained country rows are imputed using column medians, followed by zero-mean unit-variance standard scaling (`StandardScaler`). Zero-variance features are detected and rejected.

### Clustering Algorithms
- **K-Means (Euclidean Baseline):** Implements standard Lloyd's K-Means clustering, minimizing Within-Cluster Sum of Squares (WCSS / inertia) using Euclidean distances and feature means as cluster centers.
- **K-Medians (Manhattan / Robust Alternative):** Implements genuine Manhattan (L1) clustering, minimizing the sum of absolute deviations and updating cluster centroids using feature-wise medians. Highly robust to outliers and skewed burden indicators.

### Model Evaluation
- **Elbow curves:** Plots WCSS (Inertia) for K-Means and L1 Objective (Manhattan) for K-Medians across evaluated cluster counts $k$.
- **Silhouette scores:** Computed using the algorithm's respective distance metric (Euclidean for K-Means, Manhattan for K-Medians).
- **Davies–Bouldin Index:** Reported exclusively for Euclidean K-Means, avoiding invalid Manhattan claims.
- **Safe degenerate handling:** Degenerate clusterings or singleton assignments return `NaN` gracefully rather than triggering application crashes.

### Principal Component Analysis (PCA)
- **Unified model:** A single PCA model is fitted on the standardized feature matrix ($X$) across all components.
- **2D Projection Scatter:** Visualizes country clusters on PC1 and PC2, explicitly reporting individual variance contributions on axis titles.
- **Dual-Panel Scree Plot:** Displays individual explained variance bar chart and cumulative variance curve with $90\%$ and $95\%$ reference thresholds.
- **Feature Loadings Interpretation:** Renders a grouped horizontal bar chart for $\le 2$ components or a feature $\times$ component heatmap for $\ge 3$ components. Explicitly documents loading coefficients as linear direction weights rather than correlations or causal factors.

### Distributional Cluster Profiles
- Computes Mean, Standard Deviation, Median, Q1, Q3, and Interquartile Range (IQR) on the **original (unscaled)** feature values to account for skewed health distributions.

### Interactive State Persistence & Downloads
- Experiment results, figures, and metrics persist in `st.session_state` across dropdown selections and download clicks.
- Automatic cache invalidation ensures stale results clear upon file, feature, missingness threshold, or algorithm changes.
- Three dedicated CSV download channels:
  1. Country cluster assignments (`country_clusters_k{selected_k}.csv`).
  2. Model evaluation metrics (`clustering_evaluation.csv`).
  3. Excluded countries audit log (`excluded_countries_audit.csv`).

---

## 3. Technology Stack & Prerequisites
- **Python:** 3.11+ / 3.12 (Tested on Python 3.12.4)
- **Core Libraries:**
  - `streamlit>=1.39,<2.0`
  - `pandas>=2.2,<3.0`
  - `numpy>=1.26,<3.0`
  - `scikit-learn>=1.5,<2.0`
  - `plotly>=5.24,<7.0`
  - `pytest>=8.0,<9.0`

---

## 4. Local Installation & Setup

### Clone and Environment Creation
```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r project28_starter/requirements.txt
```

### Running the Application
From the workspace root:
```bash
streamlit run project28_starter/app.py
```
Or from inside the `project28_starter` directory:
```bash
cd project28_starter
streamlit run app.py
```

### Running the Test Suite
From the workspace root:
```bash
pytest project28_starter/tests -v
```
Or execute with the specific virtual environment binary:
```bash
& ".venv\Scripts\python.exe" -m pytest project28_starter/tests -v
```

---

## 5. Input Data Contract & Synthetic Example

The uploaded CSV file must contain at least 3 country rows and at least 2 usable numeric feature columns.

### Synthetic Example CSV
> [!NOTE]
> The data below is **purely synthetic** and generated for testing and demonstration purposes. It does not represent actual IHME/GBD observations.

```csv
country,cardiovascular_death_rate,neoplasm_death_rate,respiratory_daly_rate
Country_Alpha,245.3,180.2,45.1
Country_Beta,412.8,110.5,82.4
Country_Gamma,189.4,220.7,31.2
Country_Delta,320.1,145.0,60.8
Country_Epsilon,150.2,260.4,25.9
```

### Input Requirements & Missingness Policy
1. **Country Column:** Must be named `country`, `location`, `entity`, `iso3`, etc. Every row must have a non-blank, unique value.
2. **Numeric Features:** Must contain numeric values (rates per 100k, DALYs, percentages).
3. **Missingness:** Missing cells (`NaN`) are tolerated up to the configured slider threshold ($30\%$ default). Rows exceeding the threshold are excluded and logged in the downloadable audit CSV. Remaining missing cells are imputed with column medians.

---

## 6. Interpreting Application Visualizations

1. **Evaluation Curves:**
   - **Elbow Curve (WCSS / L1 Objective):** Look for a distinct leveling-off or "elbow" where adding another cluster yields diminishing returns in variance reduction.
   - **Silhouette Score:** Ranges from $-1$ to $+1$. Higher values indicate better-separated and more cohesive clusters. Scores above $0.5$ generally indicate reasonable structure.
   - **Davies–Bouldin Index (K-Means only):** Lower values indicate better clustering (tighter clusters with greater separation).
2. **PCA Projection Scatter:**
   - Shows observations projected onto the 2 directions of maximum variance (PC1 and PC2).
   - Proximity indicates overall multivariate similarity across standardized features. Note that clustering was computed on all selected features, not exclusively on these 2 dimensions.
3. **Scree Plot:**
   - Left panel shows individual percentage of variance captured by each component.
   - Right panel shows cumulative variance. Red reference lines at $90\%$ and $95\%$ indicate how many components are required to retain the majority of information.
4. **PCA Loadings:**
   - Bar chart ($\le 2$ components) or Heatmap ($\ge 3$ components) shows feature weights. Positive/negative values indicate the relative contribution of each standardized feature to the principal component axis.

---

## 7. Known Limitations

- **Ecological & Cross-Sectional:** Country-level aggregates mask within-country inequalities, urban-rural disparities, and socioeconomic gradients.
- **Harmonization Dependencies:** Comparing indicators across differing collection methodologies, years, or registration completeness requires external domain harmonization.
- **Heuristic Cluster Selection:** Unsupervised metrics (Silhouette, Elbow) provide mathematical heuristics, not clinical ground truth.
- **Pending Benchmark Integration:** The official IHME/GBD benchmark dataset has not yet been integrated and must be obtained under authorized educational terms.

---

## 8. Deployment Preparation (Streamlit Community Cloud)

- **Entry Point:** `project28_starter/app.py`
- **Dependencies:** `project28_starter/requirements.txt`
- **Configuration:** `.streamlit/config.toml` pre-configured with `headless = true` and `maxUploadSize = 25`.
- **Portability:** Code uses relative module imports (`src.*`) and path-agnostic file handling, with zero hardcoded local drive paths. No GPU or external database is required.
