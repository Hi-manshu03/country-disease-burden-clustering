# Testing and release checklist

- [x] `pip install -r requirements.txt` succeeds from a clean environment (`pip check` verified 0 broken requirements).
- [x] `pytest project28_starter/tests -v` passes (107/107 tests passed).
- [x] App starts with `streamlit run app.py` (entry point `project28_starter/app.py`).
- [x] Invalid CSV shows a useful error (missing country identifier, blank names, duplicates, <2 numeric features, all-missing columns, invalid k).
- [x] Missing values are handled and reported (configurable threshold slider, exclusion audit log, and median imputation).
- [x] Constant columns are rejected with a useful message.
- [x] At least three k values can be evaluated on the chosen dataset.
- [x] WCSS, Silhouette, and Davies–Bouldin metrics are displayed (WCSS and DB index for K-Means; L1 Objective and Manhattan Silhouette for K-Medians).
- [x] PCA plot has country hover labels, cluster colors, and individual axis variance ratios.
- [x] Scree plot and PCA loadings (bar/heatmap) derive from single fitted model and match variance ratios.
- [x] Cluster assignments, metrics, and exclusion audit download correctly and persist in session state.
- [x] Results are reproducible with fixed random seed (`random_state=42`).
- [ ] Real IHME dataset provenance and license documented (planned under P28-008 upon authorized download).
- [ ] Deployed URL loads and completes one end-to-end run (planned under final cloud deployment).
