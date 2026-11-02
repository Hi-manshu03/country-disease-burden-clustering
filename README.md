# Country Disease-Burden Pattern Clustering

Project 28 is an interactive Streamlit dashboard for exploring statistical similarities between countries using comparable disease-burden indicators from the IHME Global Burden of Disease (GBD) study. It supports reproducible clustering experiments, internal evaluation, PCA visualization, interpretable cluster summaries and data exports.

> **Responsible interpretation:** Clusters describe patterns in the selected data. They do not establish causality, provide diagnoses, or rank countries or health systems.

## Dashboard sections

The dark-only interface uses the supplied amber/graphite visual system and background asset. The tabs separate:

1. **Overview** — analytical workflow, data-use context and responsible interpretation.
2. **Dataset & Ingestion** — upload prepared country-level CSVs or GBD long-format exports; inspect data and metadata.
3. **Algorithm & Configuration** — feature selection, missingness policy, cluster counts, random seed and optional paired algorithm comparison.
4. **Model Evaluation** — objective curves, Silhouette scores, Davies–Bouldin Index where defined, and metric interpretation.
5. **Dimensionality Reduction** — PCA projection, explained variance, scree plot and component loadings.
6. **Cluster Profiles** — original-scale distribution statistics, country search and cluster filtering.
7. **Reports & Export** — assignment, profile, metric, exclusion-audit CSVs and a plain-text analysis report.

## Data and provenance

The dashboard accepts either:

- A prepared CSV with one unique country/territory per row and at least two usable numeric feature columns; or
- An IHME GBD long-format CSV containing `location_name`, `cause_name`, and `val`.

GBD input is filtered to DALYs, Rate, Both sexes and Age-standardized records when those dimensions are present. A year is selected explicitly when available. Duplicate location/cause estimates are rejected rather than averaged. See [data provenance](docs/data_provenance.md) and the [feature dictionary](docs/feature_dictionary.md).

The supplied local extract is GBD 2023 data with 3,060 long-format records, 204 locations and 15 selected causes. It transforms into a 204-country × 15-feature matrix. The raw and processed data files are excluded from Git; they are not bundled for cloud deployment. Confirm IHME terms before redistribution. The download date was not recorded.

### Input quality and preprocessing

- Country names/codes must be present, nonblank and unique.
- Numeric strings are converted; invalid and infinite values are treated as missing.
- The user-configured row missingness threshold defaults to 30%. Excluded countries are listed in an audit table.
- Remaining missing feature cells are median-imputed.
- Zero-variance features are rejected.
- Standard scaling is used for clustering and PCA; cluster profiles report original, unscaled values.
- A fixed random seed (default 42) makes experiments reproducible.

## Methods

- **K-Means:** Euclidean assignment and mean-based centroids; the objective is WCSS (sum of squared distances).
- **K-Medians:** Manhattan assignment and component-wise median representatives; the objective is the sum of L1 distances.
- **Evaluation:** Silhouette is computed using the selected algorithm's distance metric. Davies–Bouldin is reported for K-Means only. Objective values from the two algorithms have different definitions and are not directly comparable.
- **PCA:** A single fitted model on standardized features provides the scatter plot, variance summary and loadings. Clustering itself uses all selected features, not only the two displayed components.
- **Profiles:** Mean, standard deviation, median, quartiles and IQR are computed on retained original-scale feature values.

No single internal metric determines an optimal cluster count. Interpret evaluation metrics jointly and in light of domain knowledge.

## Architecture

```text
app.py                    Streamlit UI, navigation, session state and exports
src/data.py               Dataset validation, missingness handling and scaling
src/gbd_transform.py      GBD long-to-wide filtering and validation
src/clustering.py         K-Means, K-Medians, evaluation and cluster profiles
src/visualization.py      PCA, evaluation and profile charts
assets/style.css          Dark dashboard styles and responsive layout
static/dashboard-bg.png   Dashboard background, served by Streamlit
tests/                    Unit and Streamlit AppTest coverage
docs/                     Provenance, feature and testing documentation
```

## Installation and execution

Python 3.11 or newer is recommended.

```powershell
cd project28_starter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

From the project directory, run the tests with:

```powershell
python -m pytest tests -q
```

## Deployment preparation

The project entry point is `app.py`; dependency and Streamlit configuration are in `requirements.txt` and `.streamlit/config.toml`. The background asset is served from the project `static/` directory. For Streamlit Community Cloud, point the deployment at this directory and ensure the repository contains the application files and asset.

The GBD raw and processed files are intentionally ignored by Git. A deployed app therefore requires users to upload a dataset unless the project owner separately authorizes and provisions a distributable dataset. No public deployment or Git push has been performed.

## Limitations

- Country-level data conceal within-country and subnational differences.
- GBD estimates are subject to uncertainty, modeling assumptions and differences in source-data completeness.
- Cross-country comparability depends on consistent source years, measures, units, age groups and sexes.
- Internal clustering metrics are heuristics, not external clinical or epidemiological validation.
- The dashboard performs descriptive analysis; it does not support causal, diagnostic or policy-effectiveness claims.

For further test and release status, see [testing documentation](docs/testing.md).
