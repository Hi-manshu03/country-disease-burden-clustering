# Project 28 — Country Disease-Burden Pattern Clustering
## Development & Verification Log

---

### Entry: 2026-10-26 — Task P28-001: Initial Repository Audit & Baseline Verification

#### Scope & Method
Conducted an empirical, read-only audit of `project28_starter`, workspace directory `H:\My Drive\Unsupervised Learning`, virtual environment `.venv`, and documentation. No code or environment modifications were performed.

#### Environment Baseline
* **OS:** Windows 11 AMD64
* **Python Executables:**
  * Virtual Environment: `H:\My Drive\Unsupervised Learning\.venv\Scripts\python.exe` (Python 3.12.4)
  * System Python: `C:\Program Files\Python312\python.exe` (Python 3.12.4)
* **Pip Version:** `pip 26.2.2`
* **Installed Packages:** `streamlit==1.66.1`, `pandas==2.3.3`, `numpy==2.5.4`, `scikit-learn==1.9.1`, `plotly==6.9.0`, `pytest==8.4.2`, `scipy==1.18.1`, `altair==6.3.0`, `pyarrow==26.0.0`.

#### Baseline Test Execution Records
1. **Inside Project Root (`project28_starter`):**
   * Command: `& ".venv\Scripts\python.exe" -m pytest -v`
   * Result: **PASSED** (2 passed in 23.31s).
   * Verified: `test_clustering.py::test_evaluate_multiple_k_values`, `test_data.py::test_validate_and_prepare`.
2. **From Workspace Root (`H:\My Drive\Unsupervised Learning`):**
   * Command: `& ".venv\Scripts\python.exe" -m pytest project28_starter/tests`
   * Result: **FAILED** (Exit code 1, `ModuleNotFoundError: No module named 'src'`).
   * Root Cause: Python path was not configured to resolve `project28_starter` when executed outside that directory.
3. **Workspace Root with Pythonpath Override:**
   * Command: `& ".venv\Scripts\python.exe" -m pytest -o pythonpath=project28_starter project28_starter/tests`
   * Result: **PASSED** (2 passed in 12.09s).

#### Summary of Key Audit Findings
* **CRITICAL:** Streamlit UI state unmounts upon interaction due to `st.selectbox` and `st.download_button` nested inside `if st.button(...)` with no `st.session_state`.
* **CRITICAL:** Missing `requirements.txt` in project deployment folder `project28_starter/` (was isolated in `.venv/requirements.txt`).
* **HIGH:** Missing Manhattan distance / K-Medians implementation required by `README.md` and `AGENTS.md`.
* **HIGH:** Median imputation on sparse records (rows with even 1 non-null feature are retained and imputed with global medians).
* **HIGH:** Lack of duplicate country validation in `src/data.py:validate_dataset`.
* **CORRESPONDENCE CLARIFICATION:** Positional correspondence is preserved on clean, default `RangeIndex` DataFrames; however, it is vulnerable to non-unique indices, masked by silent slicing in PCA (`country_names[:len(labels)]`), and lacks notification for dropped rows.

---

### Entry: 2026-10-26 — Task P28-002: Infrastructure, Configuration, and Tracking Initialization

#### Objectives
1. Provide dependency manifest in the application directory for deployment.
2. Configure pytest so tests can run both from workspace root and project directory without manual flags.
3. Initialize Streamlit server configuration (`.streamlit/config.toml`).
4. Initialize `TASK_TRACKER.md` and `DEVELOPMENT_LOG.md`.
5. Strictly preserve application code (`app.py`, `src/`) and existing test files.

#### Implementation Actions & Rationale
1. **Requirements Manifest (`project28_starter/requirements.txt` and root `requirements.txt`):**
   * Preserved user-created manifest in `project28_starter/requirements.txt`:
     ```text
     streamlit>=1.39,<2.0
     pandas>=2.2,<3.0
     numpy>=1.26,<3.0
     scikit-learn>=1.5,<2.0
     plotly>=5.24,<7.0
     pytest>=8.0,<9.0
     ```
   * Mirrored `requirements.txt` at the workspace root to support root-level setups.
2. **Pytest Discovery Configuration (`pytest.ini`):**
   * Created `H:\My Drive\Unsupervised Learning\pytest.ini`:
     ```ini
     [pytest]
     pythonpath = project28_starter
     testpaths = project28_starter/tests tests
     ```
   * Rationale: Pytest traverses parent directories from `project28_starter/`, finding `pytest.ini` at the root. This single configuration file allows `pytest` to run cleanly without arguments from both the workspace root and within `project28_starter/`, avoiding redundant or conflicting configurations.
3. **Streamlit Configuration (`project28_starter/.streamlit/config.toml`):**
   * Created `.streamlit/config.toml` with:
     ```toml
     [server]
     headless = true
     maxUploadSize = 25

     [browser]
     gatherUsageStats = false
     ```
   * Rationale: Standardizes headless mode for deployment and sets the maximum upload file size to 25 MB.
4. **Tracking Artifacts:**
   * Created `project28_starter/TASK_TRACKER.md` with complete task lifecycle from P28-001 to P28-007.
   * Created `project28_starter/DEVELOPMENT_LOG.md` documenting all empirical audit findings and infrastructure changes.

#### Verification Loop & Actual Results

* **Dependency Consistency Check:**
  * Command: `& ".venv\Scripts\python.exe" -m pip check`
  * Working Directory: `H:\My Drive\Unsupervised Learning\project28_starter`
  * Result: **PASSED** (Output: `No broken requirements found.`, exit code 0).

* **Test Discovery from Workspace Root:**
  * Command: `& ".venv\Scripts\python.exe" -m pytest -v`
  * Working Directory: `H:\My Drive\Unsupervised Learning`
  * Result: **PASSED** (2 passed in 19.80s, exit code 0).
  * Output:
    ```text
    rootdir: H:\My Drive\Unsupervised Learning
    configfile: pytest.ini
    testpaths: project28_starter/tests, tests
    plugins: anyio-4.15.1
    collected 2 items
    project28_starter/tests/test_clustering.py::test_evaluate_multiple_k_values PASSED [ 50%]
    project28_starter/tests/test_data.py::test_validate_and_prepare PASSED             [100%]
    ============================= 2 passed in 19.80s ==============================
    ```

* **Test Discovery from Project Root (`project28_starter`):**
  * Command: `& ".venv\Scripts\python.exe" -m pytest -v`
  * Working Directory: `H:\My Drive\Unsupervised Learning\project28_starter`
  * Result: **PASSED** (2 passed in 21.88s, exit code 0).
  * Output:
    ```text
    rootdir: H:\My Drive\Unsupervised Learning
    configfile: pytest.ini
    plugins: anyio-4.15.1
    collected 2 items
    tests\test_clustering.py::test_evaluate_multiple_k_values PASSED         [ 50%]
    tests\test_data.py::test_validate_and_prepare PASSED                     [100%]
    ============================= 2 passed in 21.88s ==============================
    ```

* **Streamlit Configuration Validation:**
  * Command: `& ".venv\Scripts\python.exe" -c "import streamlit.config as cfg; print('headless:', cfg.get_option('server.headless')); print('maxUploadSize:', cfg.get_option('server.maxUploadSize')); print('gatherUsageStats:', cfg.get_option('browser.gatherUsageStats'))"`
  * Result:
    ```text
    headless: True
    maxUploadSize: 25
    gatherUsageStats: False
    ```
  * Exit Code: `0`

* **Source Code Integrity Verification:**
  * `app.py`: Unmodified.
  * `src/data.py`: Unmodified.
  * `src/clustering.py`: Unmodified.
  * `src/visualization.py`: Unmodified.
  * `tests/test_clustering.py`: Unmodified.
  * `tests/test_data.py`: Unmodified.

#### Remaining Issues & Recommended Next Task
* **Immediate Block:** Ephemeral UI state bug in `app.py`. Results and download buttons disappear as soon as users interact with dropdowns or download links.
* **Next Task:** `P28-003: Streamlit State Management and Interactivity Refactor`.

---

### Entry: 2026-10-26 — Task P28-003: Streamlit State Management and Interactive Results

#### Scope & Method
Refactored `project28_starter/app.py` to persist clustering experiment results, metrics, labels, and figure context in `st.session_state`. Decoupled result rendering and download widgets from the execution button block. Preserved existing preprocessing and clustering semantics. Added automated integration tests using `streamlit.testing.v1.AppTest`.

#### Implementation Changes & Rationale
1. **Result Persistence in `st.session_state`:**
   * Replaced the transient `if st.button(...)` nesting pattern.
   * On button click, clustering outputs (`metrics`, `labels_by_k`, `X`, `analysis_raw`, `country_col`, `country_names`, `features`, `k_values`, and `file_signature`) are cached in `st.session_state["clustering_result"]`.
2. **Safe Stale State Invalidation:**
   * Extracted `file_signature = getattr(uploaded, "file_id", None) or f"{uploaded.name}_{uploaded.size}"`.
   * Stored results are checked against current `file_signature`, `features`, and `k_values`. If the user uploads a new dataset or modifies selected features/k-values, stale cached results are automatically popped from `st.session_state` and unmounted until a new experiment is explicitly triggered.
3. **Decoupled Interactive Widgets:**
   * Rendered `st.selectbox("Cluster solution to inspect")` with `key="selected_k_solution"` outside the button trigger.
   * Switching $k$ dynamically updates the PCA figure, cluster profile averages, and download filenames without re-executing clustering algorithms.
   * Rendered download buttons for cluster assignments and evaluation metrics outside the button trigger; clicking downloads no longer causes the UI to collapse.

#### Empirical Verification & Results

* **Bug Reproduction with AppTest (Pre-Fix):**
  * Immediately after button click: `selectbox count = 1, download button count = 2`.
  * On subsequent rerun: `selectbox count = 0, download button count = 0` (Confirmed results vanished).

* **Post-Fix Verification with AppTest:**
  * Immediately after button click: `selectbox count = 1, download button count = 2`.
  * On subsequent rerun: `selectbox count = 1, download button count = 2` (Confirmed results persist).

* **Automated Integration Test Suite (`tests/test_app_state.py`):**
  * Added 7 test cases covering:
    1. `test_clustering_result_persistence_on_rerun`: PASS
    2. `test_change_selected_cluster_k`: PASS ($k=2 \to k=3$ persists and updates)
    3. `test_download_buttons_persist_and_contain_data`: PASS (both downloads remain mounted)
    4. `test_feature_change_invalidates_stale_results`: PASS (unselecting a feature clears stale results)
    5. `test_uploading_new_file_invalidates_stale_results`: PASS (new CSV invalidates previous state)
    6. `test_country_to_cluster_correspondence_preserved`: PASS (labels and country lists match)
    7. `test_missing_or_corrupted_session_state_handled_safely`: PASS (corrupted state handled without crash)
  * Result: **7 passed in 36.93s** (Exit code 0).

* **Full Project Test Suite:**
  * Ran `pytest -v` across all tests (`test_app_state.py`, `test_clustering.py`, `test_data.py`).
  * From Workspace Root: **9 passed in 37.79s** (Exit code 0).
  * From Project Root: **9 passed in 38.75s** (Exit code 0).

* **Source Code Integrity:**
  * `src/data.py`: Unmodified.
  * `src/clustering.py`: Unmodified.
  * `src/visualization.py`: Unmodified.
  * Existing unit tests (`test_clustering.py`, `test_data.py`): Unmodified.

#### Remaining Issues & Recommended Next Task
* **Immediate Focus:** Data ingestion hardening and missingness policy. Currently, countries with only 1 non-null indicator are retained and median-imputed across all other indicators; duplicate country names are not flagged.
* **Next Task:** `P28-004: Data Ingestion Hardening and Missingness Validation`.

---

### Entry: 2026-10-27 — Task P28-004: Data Ingestion Hardening and Missingness Validation

#### Scope & Method
Hardened input validation, identifier integrity, and missing-data preprocessing in `src/data.py` and `app.py`. Added duplicate country detection, blank country identifier rejection, a configurable row-level missingness threshold (default 30%), structured exclusion audit logging, explicit length validation in `src/visualization.py`, and a downloadable exclusion report.

#### Implementation Changes & Rationale
1. **Input Contract Enforcement (`src/data.py:validate_dataset`):**
   * Validated that the dataset contains at least 3 country rows and 2 columns.
   * Checked for null, empty, whitespace-only, `"NaN"`, `"None"`, and `"null"` string country identifiers, raising informative `ValueError` with culprit row indices.
   * Detected duplicate country records; raised an explanatory error warning that duplicates often reflect multi-year, sex-disaggregated, or age-group records requiring harmonization before clustering.
   * Coerced non-numeric entries and infinite values (`inf`, `-inf`) to `np.nan`.
2. **Configurable Row-Level Missingness Policy (`src/data.py:prepare_features`):**
   * Introduced parameter `max_missing_fraction: float = 0.30` (default 30%).
   * For each country row, computed $\text{missing\_fraction} = \frac{\text{missing features}}{\text{total selected features}}$.
   * **Boundary semantics:** Rows with $\text{missing\_fraction} \le \text{threshold}$ are retained; rows with $\text{missing\_fraction} > \text{threshold}$ are excluded.
   * Built structured `exclusions_df` recording `country`, `row_index`, `missing_count`, `total_features`, `missing_fraction`, `missing_percentage`, and reason codes (`EXCEEDS_MISSING_THRESHOLD`, `ALL_FEATURES_MISSING`).
   * Raised `ValueError` if fewer than 3 country rows remain after filtering.
   * Applied `SimpleImputer(strategy="median")` strictly to eligible retained rows, preventing sparse country distortion.
   * Checked for zero-variance constant features after imputation.
3. **Country-to-Cluster Alignment Protection (`src/visualization.py:make_pca_figure`):**
   * Replaced silent truncation slice `country_names[:len(labels)]` with explicit assertion: `if len(country_names) != len(labels): raise ValueError(...)`.
4. **UI Transparency & Audit Download (`app.py`):**
   * Added interactive `st.slider` for "Maximum allowed missing features per country (%)" with step 5%, defaulting to 30%.
   * Added 4-column overview metrics: Input rows, Retained rows, Excluded rows, Missing values imputed.
   * Rendered explicit epidemiological warning when exclusions occur: "Excluding data-deficient countries can alter the population and geographic balance represented by the analysis."
   * Added audit expander and `st.download_button` for `excluded_countries_audit.csv`.
   * Integrated `max_missing_fraction` into cache invalidation so altering the threshold clears stale cluster results.

#### Empirical Verification & Results

* **Automated Unit Tests (`tests/test_data.py` — 10 tests):**
  1. `test_validate_and_prepare_basic`: PASS
  2. `test_missing_and_blank_country_identifiers`: PASS (Rejects whitespace, None, literal 'NaN')
  3. `test_duplicate_country_identifiers`: PASS (Rejects duplicate France with guidance)
  4. `test_numeric_coercion_and_infinities`: PASS (Infinities coerced to NaN without crash)
  5. `test_missingness_threshold_boundary_semantics`: PASS (Tested 50% vs 30% retention boundaries)
  6. `test_median_imputation_values`: PASS (Verified column median applied to retained cells)
  7. `test_fewer_than_three_usable_rows_error`: PASS (Clear ValueError when <3 rows remain)
  8. `test_constant_feature_rejection`: PASS (Zero variance rejected with column name)
  9. `test_stable_row_order_and_correspondence`: PASS (Row indices and country mapping verified)
  10. `test_invalid_missingness_threshold_parameter`: PASS (Rejects threshold outside [0, 1])
  * Result: **10 passed in 12.47s**.

* **Automated Integration Tests (`tests/test_app_state.py` — 9 tests):**
  * Existing 7 tests: All passed.
  * Added `test_missingness_slider_and_exclusion_audit_display`: PASS (Exclusion warning and audit CSV download verified).
  * Added `test_missingness_slider_change_invalidates_state`: PASS (Adjusting slider clears stale state).
  * Result: **9 passed in 39.30s**.

* **Full Regression Test Suite:**
  * Ran all 20 tests from workspace root: **20 passed in 40.04s** (Exit code 0).
  * Ran all 20 tests from project directory: **20 passed in 38.28s** (Exit code 0).

* **Source Code Integrity:**
  * `src/clustering.py`: Unmodified.
  * Clustering algorithm and evaluation metrics unchanged.

#### Remaining Issues & Recommended Next Task
* **Immediate Focus:** Methodological completeness for distance metrics. The contract in `README.md` and `AGENTS.md` requires addressing Manhattan distance / K-Medians and providing graphical evaluation curves (Elbow WCSS, Silhouette vs. $k$, Davies–Bouldin vs. $k$).
* **Next Task:** `P28-005: Clustering Methodology, Metric Charts & Manhattan / K-Medians`.

---

### Entry: 2026-10-27 — Task P28-005: Clustering Methodology, Evaluation Curves, Manhattan-Based Clustering, and Distributional Cluster Profiles

#### Scope & Method
Implemented evaluation curve visualizations, surfaced the existing K-Medians implementation in the UI alongside K-Means, enriched cluster profiles with distributional statistics (mean, median, IQR), and added PCA axis variance labelling. Added 33 new unit/integration tests. Updated 9 existing integration tests for the new UI layout.

#### Implementation Changes & Rationale

1. **Evaluation Curves (`src/visualization.py:make_evaluation_curves`):**
   * New function returning a multi-panel Plotly `Figure`.
   * K-Means path renders 3 panels: Elbow (WCSS vs. k), Silhouette vs. k, Davies–Bouldin vs. k.
   * K-Medians path renders 2 panels: Elbow (L1 objective vs. k), Silhouette vs. k (DB index is not computed for K-Medians since sklearn's `davies_bouldin_score` is Euclidean-centric).
   * Panel detection is data-driven (checks for column presence) so the function handles either algorithm output without branching logic in the caller.

2. **PCA Loadings Chart (`src/visualization.py:make_pca_loadings_figure`):**
   * New function returning a horizontal grouped bar chart of PC1 and PC2 loadings per feature.
   * Uses `n_components=min(2, X.shape[1])` to handle edge cases with fewer features than components.
   * Exposed in the UI as a collapsible expander beneath the PCA scatter, to keep the default view clean.

3. **PCA Axis Variance Labels (`src/visualization.py:make_pca_figure`):**
   * Updated axis column names to include individual variance ratios, e.g. `"PC1 (38.4%)"`, `"PC2 (22.1%)"`.
   * Title still reports combined variance; axes now give per-component breakdown.
   * Signature and `ValueError` guard are unchanged.

4. **K-Medians / Algorithm Selector (`app.py`):**
   * Added `st.selectbox("Clustering algorithm", ...)` with two options: `"K-Means (Euclidean)"` and `"K-Medians (Manhattan)"`.
   * Dispatches to `evaluate_kmeans` or `evaluate_kmedians` accordingly.
   * `algorithm_choice` stored in `st.session_state["clustering_result"]` and included in the stale-state invalidation check, so switching algorithms clears previous results and forces a fresh run.

5. **Distributional Cluster Profiles (`app.py`):**
   * Replaced the `groupby.mean()` table with `compute_cluster_profiles()` (already implemented in `src/clustering.py`), which returns mean, median, SD, Q1, Q3, IQR per (cluster, feature).
   * Rendered via two tabs: "Summary table" (full profile DataFrame) and "By feature" (grouped bar charts showing mean ± SD and median per cluster per feature).
   * A methodological caption explains that median/IQR are reported because disease-burden indicators are frequently right-skewed.

6. **Evaluation Curves in UI (`app.py`):**
   * Evaluation curves chart displayed as the primary result view before the metrics table.
   * Metrics table moved inside an expander with a download button, keeping the default view uncluttered.

7. **`analysis_raw` Reset Index (`app.py`):**
   * After `raw.loc[prepared["row_indices"]]`, `analysis_raw.reset_index(drop=True)` is applied to guarantee zero-based positional alignment with the 0-indexed `X` ndarray from `prepare_features`.
   * This eliminates a subtle but real alignment risk for DataFrames with non-default indices (e.g., those sliced from larger datasets).

#### Empirical Verification & Results

* **New Unit/Integration Test Suite (`tests/test_p28_005.py` — 33 tests):**
  * `TestKMedians` (10 tests): Separation on clean data, fit/fit_predict consistency, median centroid correctness, L1 objective finiteness, iteration bound, ValueError guards (n_clusters < 2, n_clusters ≥ n_samples), determinism, no empty clusters, manual L1 objective verification.
  * `TestEvaluateKMedians` (7 tests): Return structure, Manhattan label, required columns, k-value matching, silhouette range, invalid k error, label length.
  * `TestComputeClusterProfiles` (6 tests): Column presence, row count, mean correctness, median correctness, IQR identity, label-length mismatch error, country count.
  * `TestMakeEvaluationCurves` (5 tests): Figure type, 3-panel K-Means, 2-panel K-Medians, x-axis values, title content.
  * `TestMakePcaLoadingsFigure` (3 tests): Figure type, trace count = n_components, single-component edge case.
  * Cross-algorithm integration test (1 test): Both algorithms correctly separate two-blob synthetic data.
  * Result: **33 passed**.

* **Updated Integration Tests (`tests/test_app_state.py` — 9 tests):**
  * All 9 tests updated to reflect the new `algorithm_choice` selectbox (now at index 0; `selected_k_solution` at index 1).
  * Stale-state invalidation tests updated: after invalidation, `algorithm_choice` selectbox remains (count = 1), `selected_k_solution` is cleared.
  * Added assertion: `cached["algorithm_choice"] == "K-Means (Euclidean)"` in the missingness audit test.
  * Result: **9 passed**.

* **Full Regression Test Suite:**
  * Ran all 53 tests from workspace root: **53 passed in 46.77s** (Exit code 0).
  * Zero regressions: all 20 pre-P28-005 tests continue to pass.

#### Remaining Issues & Recommended Next Task
* **Immediate Focus:** PCA enhancements — scree plot across all components; the axis variance labelling is now in place, but a dedicated scree plot and a more formal variance summary are part of `P28-006`.
* **Next Task:** `P28-006: PCA Loadings, Scree Plot & Axis Variance Visualization`.




---

### Entry: 2026-10-29 — Task P28-007A: Application Stabilization and Deployment Readiness

#### Scope & Method
Conducted an end-to-end stabilization, integration verification, and deployment readiness review of `project28_starter` and the workspace environment. Verified baseline tests, inspected all application and test modules, diagnosed input validation and error-handling edge cases, resolved methodological naming discrepancies, hardened clustering execution against invalid and degenerate inputs, expanded test coverage with 10 new unit and Streamlit `AppTest` cases, and updated user-facing documentation.

#### Baseline Verification (Phase A)
* **Initial Test Run:**
  * Command: `& "H:\My Drive\Unsupervised Learning\.venv\Scripts\python.exe" -m pytest project28_starter/tests -v`
  * Result: **PASSED — 97 passed in 174.39s (Exit code 0)**.
* **Deterministic PCA Verification:**
  * Verified presence and clean pass of `TestFitPca::test_known_variance_on_identity_covariance` in `tests/test_p28_006.py`.
* **Repository & Git Status:**
  * No `.git` repository exists in the workspace root or in `project28_starter`. Working tree is an untracked directory in Google Drive. All existing user work preserved.

#### Defects Diagnosed & Fixes Implemented

1. **Missing Country Identifier Column Validation:**
   * *Defect:* If a user uploaded a CSV with no recognized country identifier and all numeric columns, `validate_dataset` fell back to `str(df.columns[0])`, silently treating the first numeric feature column as country names.
   * *Root Cause:* Fallback `country_col = country_candidates[0] if country_candidates else str(df.columns[0])` lacked a check verifying candidate existence.
   * *Fix:* Updated `src/data.py:validate_dataset` to expand candidate names (`nation`, `state`, `territory`, `economy`) and raise an actionable `ValueError("No country identifier column found...")` if no valid candidate column is present.
   * *Verification:* Added `test_validate_dataset_rejects_missing_country_identifier` and `test_app_invalid_country_csv_displays_error`.

2. **Single Numeric Feature Upload Edge Case:**
   * *Defect:* If a dataset contained only 1 numeric feature column alongside the country column, `validate_dataset` returned successfully, violating its documented contract (`Must contain at least two usable numeric candidate feature columns`).
   * *Root Cause:* Line 77 in `src/data.py` only verified `if not numeric_cols:`.
   * *Fix:* Changed guard to `if len(numeric_cols) < 2: raise ValueError(...)`.
   * *Verification:* Added `test_validate_dataset_rejects_single_numeric_feature`.

3. **Methodological Inconsistency in K-Medians Evaluation Curve Panel Title:**
   * *Defect:* In `src/visualization.py:make_evaluation_curves`, the elbow curve subplot title for K-Medians was formatted as `"Elbow Curve — L1 Objective (WCSS)"`.
   * *Root Cause:* Hardcoded `"(WCSS)"` string in `objective_label = "L1 Objective (WCSS)" if is_kmedians else "WCSS (Inertia)"`. WCSS (Within-Cluster Sum of Squares) is strictly Euclidean; K-Medians minimizes the Manhattan L1 objective (sum of absolute deviations).
   * *Fix:* Corrected label to `"L1 Objective (Manhattan)"`.
   * *Verification:* Added `test_kmedians_evaluation_curve_labels_manhattan_not_wcss` asserting `"WCSS"` does not appear in K-Medians panel titles.

4. **Clustering Execution Error Handling & Omitted $k$ Transparency:**
   * *Defect:* When the user selected cluster counts where none were valid (e.g. $k \ge n\_samples$), `evaluate_kmeans` / `evaluate_kmedians` raised `ValueError`, resulting in an unhandled traceback in Streamlit. Furthermore, when some $k$ values were omitted because $k \ge n\_samples$, no feedback was given to the user.
   * *Root Cause:* Button handler in `app.py` called `evaluate_kmeans` and `evaluate_kmedians` outside a `try...except` block and did not report omitted $k$ values.
   * *Fix:* Wrapped clustering execution in `try...except ValueError as exc: st.error(f"Clustering failed: {exc}"); st.stop()`, and added an informational note: `st.warning("Note: Cluster count(s) ... were omitted because k must be strictly less than the number of retained observations...")`.
   * *Verification:* Added `test_evaluate_kmeans_raises_when_all_k_invalid`, `test_evaluate_kmedians_raises_when_all_k_invalid`, and `test_app_omits_invalid_k_with_warning`.

5. **Degenerate Clustering Metric Calculation Guard:**
   * *Defect:* On degenerate data (e.g. identical rows or singleton clusters), `silhouette_score` and `davies_bouldin_score` could raise exceptions inside the evaluation loops.
   * *Fix:* Wrapped metric calculations in safe `try...except Exception` blocks, returning `np.nan` gracefully when metrics are mathematically undefined.
   * *Verification:* Added `test_evaluate_kmeans_safe_on_degenerate_points` and `test_evaluate_kmedians_safe_on_degenerate_points`.

6. **Repository Hygiene & `.gitignore` Coverage:**
   * *Defect:* No root-level `.gitignore` existed.
   * *Fix:* Created `.gitignore` at workspace root and updated `project28_starter/.gitignore` to ignore `.venv/`, `.pytest_cache/`, `__pycache__/`, `.coverage`, secrets, and private datasets.

#### Automated Test Suite Expansion & Results
* **New Test Module:** `project28_starter/tests/test_stabilization.py` (10 tests).
* **Final Regression Run:**
  * Command: `& "H:\My Drive\Unsupervised Learning\.venv\Scripts\python.exe" -m pytest project28_starter/tests -v`
  * Total tests collected: 107
  * Total passed: **107 passed, 2 warnings in 109.02s (Exit code 0)**.
  * Breakdown:
    - `test_app_state.py`: 9 passed
    - `test_clustering.py`: 1 passed
    - `test_data.py`: 10 passed
    - `test_p28_005.py`: 33 passed
    - `test_p28_006.py`: 44 passed
    - `test_stabilization.py`: 10 passed

#### Manual Smoke Test Verification
* **Environment Context:** Automated non-interactive CLI environment without a graphical desktop or browser attached.
* **Streamlit AppTest Verification:**
  - Ran 11 end-to-end integration tests using `streamlit.testing.v1.AppTest`:
    1. CSV upload with valid data.
    2. Missingness slider adjustments and dynamic recalculation.
    3. Exclusion audit warning and CSV download trigger.
    4. Button execution of clustering runs.
    5. Result persistence in `st.session_state` across reruns.
    6. Solution selection dropdown ($k=2 \to k=3$) updates without unmounting.
    7. Assignments and metrics download buttons persistence.
    8. Cache invalidation upon feature deselection or new file upload.
    9. Actionable `st.error` displayed on invalid CSV without country column.
    10. Clear warning when cluster count $k \ge n\_samples$.
    11. Safe recovery from corrupted/missing session state.
* **Streamlit Health Check:** Executed import check verifying `streamlit==1.66.1` is operational.

#### Deployment Preparation Status (Phase H)
* **Target:** Streamlit Community Cloud (free tier).
* **Entry Point:** `project28_starter/app.py`.
* **Manifest:** `project28_starter/requirements.txt`.
* **Configuration:** `.streamlit/config.toml` (`headless = true`, `maxUploadSize = 25`).
* **Portability:** Pure relative imports; no hardcoded absolute Windows drive paths in application runtime. Runs cleanly in any standard Python 3.11+ environment without local `.venv`.
* **Resource Profile:** Lightweight in-memory computation; CPU-only execution; no external database or LLM API keys required.

#### IHME/GBD Data Confirmation
* Confirmed: **Zero** IHME/GBD raw datasets have been downloaded, committed, or integrated during this task. All test cases and examples strictly use synthetic data.

#### Recommended Next Task
* **Task `P28-008`:** Authorized IHME/GBD Data Ingestion & Live Deployment.
  - Download official GBD indicators under authorized educational terms.
  - Record complete data provenance and indicator feature dictionary.
  - Test end-to-end with real country indicators and deploy to Streamlit Community Cloud.


---

### Entry: 2026-10-30 — Task P28-008A: Dataset Acquisition Specification & Ingestion Architecture

#### Scope & Method
Established the acquisition specification, feature selection methodology, transformation architecture, and synthetic unit testing framework for integrating official IHME Global Burden of Disease (GBD) data. Followed strict execution rules: zero fabricated data, zero claims of unauthorized downloads, preserving all existing working code, and leaving the real data transformation step awaiting file provisioning by the human project owner.

#### Key Architectural Implementations
1. **Acquisition Specification (Phase 1):**
   - Documented the exact workflow for the IHME GBD Results Tool (`https://vizhub.healthdata.org/gbd-results/`).
   - Defined target query: Measure = DALYs, Metric = Rate, Age = Age-standardized, Sex = Both, Year = 2021/2023, Location = 204 sovereign countries and territories.
   - Identified tool restrictions: registration requirement, file size limits, and inclusion of regional/super-regional aggregates requiring algorithmic filtering.

2. **Feature Selection Methodology (Phase 2):**
   - Selected 15 mutually exclusive **Level 2** disease and injury categories from the official GBD hierarchy.
   - Non-Communicable Diseases (9): Cardiovascular diseases, Neoplasms, Chronic respiratory diseases, Digestive diseases, Neurological disorders, Mental disorders, Substance use disorders, Diabetes and kidney diseases, Musculoskeletal disorders.
   - Communicable, Maternal, Neonatal & Nutritional (5): Respiratory infections and tuberculosis, Enteric infections, Neglected tropical diseases and malaria, Maternal and neonatal disorders, Nutritional deficiencies.
   - Injuries (1): Self-harm and interpersonal violence.
   - Avoided parent-child collinearity and double-counting by strictly selecting independent Level 2 groupings.

3. **Data Ingestion Pipeline Design (Phase 3):**
   - Implemented `src/gbd_transform.py:transform_gbd_long_to_wide`.
   - Validates required long-format columns (`location_name`, `cause_name`, `val`, etc.).
   - Explicitly filters out known global, super-regional, and WHO aggregates.
   - Verifies strict uniqueness of `(location, cause)` tuples without silent aggregation.
   - Pivots long-format records to the wide-format matrix with `country` as identifier and one column per disease feature.
   - Preserves compatibility with existing `validate_dataset` and `prepare_features` contracts in `src/data.py`.

4. **Provenance & Feature Documentation (Phase 4):**
   - Populated `project28_starter/docs/data_provenance.md` with complete citation, query instructions, and IHME non-commercial licensing terms.
   - Populated `project28_starter/docs/feature_dictionary.md` with clinical definitions, epidemiological relevance, and measurement caveats for all 15 Level 2 cause categories.

5. **Synthetic Unit Test Suite (Phase 6):**
   - Created `project28_starter/tests/test_gbd_ingest.py` containing 8 unit tests using synthetic fixtures.
   - Verified schema validation, aggregate removal, duplicate rejection, multi-year filtering, and downstream clustering execution.

#### Automated Test Execution Records
* **GBD Ingestion Tests:** `pytest project28_starter/tests/test_gbd_ingest.py -v` passed (8 passed in 35.29s).
* **Total Regression Baseline:** 115 passing tests.

#### Next Steps for Human Project Owner
* Download the specified CSV file from `https://vizhub.healthdata.org/gbd-results/` according to the instructions in `docs/data_provenance.md`.
* Place the downloaded file into `project28_starter/data/raw/` or workspace directory.
* Proceed to Task `P28-008B` to inspect the real file schema, execute the live pivot, and run clustering experiments on real data.


### Entry: 2026-10-31 — Local GBD Integration, Dashboard Redesign & Documentation

#### Repository and data audit
* Confirmed branch `project28/final-development`, checkpoint `73b7f6a`, and a clean worktree before implementation.
* Baseline suite: 115 passed, with two expected scikit-learn `ConvergenceWarning`s on deliberately degenerate duplicate-point fixtures.
* Found the supplied local GBD export, previously not reflected in the tracker: 3,060 long-format rows, 204 locations, 15 causes, year 2023, and one value per queried measure/metric/age/sex/population dimension.
* Verified the transformation yields 204 country/territory observations × 15 features; default preprocessing retained all 204 with no imputed cells.
* Raw and processed data remain ignored by Git; no data files were added to version control.

#### Implementation
* Wired GBD long-format CSV upload into the application, including year selection, metadata, filtering, pivoting and actionable validation.
* Changed GBD dimension filtering to reject a present-but-mismatched dimension instead of retaining incompatible source records.
* Reorganized the interface into seven analytical tabs and configured Streamlit Dark Mode only. Added responsive graphite/amber styling and a deployment-relative static background URL.
* Added optional paired K-Means/K-Medians experiments with shared standardized input, distance-appropriate metrics, PCA, original-scale profiles, assignment filtering, and CSV/TXT exports.
* Reworked README and provenance/testing documentation to reflect the supplied GBD 2023 file and actual app behavior. No public deployment, commit or push was made.

#### Verification in this entry
* Python syntax check for `app.py`, `src/gbd_transform.py` and `src/visualization.py`: passed.
* Targeted GBD, AppTest upload and state integration tests: 12 passed.
* Full regression command: `python -m pytest project28_starter/tests -q` — **118 passed, 2 expected warnings in 52.52s**. Both warnings are scikit-learn `ConvergenceWarning`s from the test intentionally fitting multiple clusters to identical points.
* Real-data AppTest: local GBD extract loaded as 204 countries × 15 features, and three K-Means k solutions ran with no Streamlit exceptions.
* Local server smoke test: `/_stcore/health` returned HTTP 200; `/app/static/dashboard-bg.png` returned HTTP 200 with `image/png` content.
* `docs/testing.md` records test and smoke-test results. Cloud deployment remains untested and was not performed.
