# Testing and release status

## Run tests

From `project28_starter/`:

```powershell
python -m pytest tests -q
```

The suite includes unit tests for input validation, missingness and scaling, K-Means and K-Medians, evaluation metrics, PCA, GBD long-to-wide transformation, plus Streamlit `AppTest` coverage for state persistence, invalidation, uploads and exports.

## Verified behavior

- Full regression command after the dashboard update: `python -m pytest tests -q` — **118 passed** in 56.65 seconds.
- The run emitted 31 Plotly deprecation warnings because the country map currently uses country-name geographic lookup, plus two expected scikit-learn `ConvergenceWarning`s from the identical-points fixture.
- Local supplied GBD 2023 extract completed an AppTest run: 204 countries, 15 features, and three evaluated k values; no Streamlit exceptions.
- Local server smoke test returned HTTP 200 from `/_stcore/health` and served `app/static/dashboard-bg.png` as `image/png`.
- Prepared CSV and GBD long-format CSV upload paths are exercised through Streamlit AppTest.
- GBD dimension mismatches are rejected; duplicate country/cause observations are not aggregated.
- Missingness, exclusion logs, feature selection, cluster-count omissions and invalid CSV errors have automated coverage.
- Clustering labels preserve country-row correspondence after filtering.
- PCA plots use a shared fitted model, and cluster profiles are computed from original feature values.
- Result state survives reruns and is invalidated when data or model configuration changes.
- The app uses a dark Streamlit theme, and its CSS references the project-local static background asset through Streamlit static serving.
- Sidebar navigation selection is covered by AppTest; the overview map and standardized cluster-profile helpers are covered for assignment/mean correctness.
- The overview evaluation-chart test verifies separate objective/silhouette/Davies–Bouldin axes, selected-K markers, and a legend positioned below the plots.
- The overview uses actual cached labels, metrics and the fitted PCA model; no separate PCA or clustering result is created for the dashboard.
- Browser smoke check: `streamlit run app.py --server.port 8502` started successfully. The supplied `Project test data/country_clusters_k2.csv` loaded as 204 observations and 16 numeric indicators; a K-Means run completed and populated all four overview panels.
- Desktop and 390px-wide mobile screenshots were reviewed. At mobile width the native sidebar collapsed with its expand control available, KPI cards stacked, and the document width remained 390px (no horizontal overflow).

## Release checks and limitations

- The project dependency manifest is `requirements.txt`; run `python -m pip check` after installing it in a clean environment.
- Automated tests and the local browser check do not replace a review across deployment browsers. Verify contrast, responsiveness and background asset loading in the target deployment before release.
- Plotly warns that the `country names` choropleth location mode may change in a future version. ISO-3 mapping is preferable when the input includes a reliable country-code field; current supported inputs may contain only country names.
- No Streamlit Community Cloud deployment or public URL has been created or tested.
- Raw and processed GBD data are intentionally ignored by Git. A deployed app requires an authorized CSV upload unless separate redistribution approval is obtained.
- The IHME source download date was not recorded with the supplied dataset.
