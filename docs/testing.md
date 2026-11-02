# Testing and release status

## Run tests

From `project28_starter/`:

```powershell
python -m pytest tests -q
```

The suite includes unit tests for input validation, missingness and scaling, K-Means and K-Medians, evaluation metrics, PCA, GBD long-to-wide transformation, plus Streamlit `AppTest` coverage for state persistence, invalidation, uploads and exports.

## Verified behavior

- Full regression command: `python -m pytest tests -q` — **118 passed** in 52.52 seconds.
- Two expected `ConvergenceWarning`s occur only in a test using identical points to verify safe degenerate-cluster handling.
- Local supplied GBD 2023 extract completed an AppTest run: 204 countries, 15 features, and three evaluated k values; no Streamlit exceptions.
- Local server smoke test returned HTTP 200 from `/_stcore/health` and served `app/static/dashboard-bg.png` as `image/png`.
- Prepared CSV and GBD long-format CSV upload paths are exercised through Streamlit AppTest.
- GBD dimension mismatches are rejected; duplicate country/cause observations are not aggregated.
- Missingness, exclusion logs, feature selection, cluster-count omissions and invalid CSV errors have automated coverage.
- Clustering labels preserve country-row correspondence after filtering.
- PCA plots use a shared fitted model, and cluster profiles are computed from original feature values.
- Result state survives reruns and is invalidated when data or model configuration changes.
- The app uses a dark Streamlit theme, and its CSS references the project-local static background asset through Streamlit static serving.

## Release checks and limitations

- The project dependency manifest is `requirements.txt`; run `python -m pip check` after installing it in a clean environment.
- Automated tests do not replace a browser-level review across devices and browsers. Verify contrast, responsiveness and background asset loading in the target deployment before release.
- No Streamlit Community Cloud deployment or public URL has been created or tested.
- Raw and processed GBD data are intentionally ignored by Git. A deployed app requires an authorized CSV upload unless separate redistribution approval is obtained.
- The IHME source download date was not recorded with the supplied dataset.
