# Project 28 — AI Agent Working Agreement

## Project goal
Deliver a correct, reproducible, free-to-deploy public-health clustering dashboard by 2026-10-31.

## Non-negotiable rules
1. Never invent IHME data, citations, measurements, or experimental results.
2. Never commit API keys, tokens, credentials, or private data.
3. Do not overwrite another agent's module without coordinating first.
4. Keep the app runnable after each integration change.
5. Every claimed feature must have a test or a documented manual verification.
6. Keep raw data out of Git unless its license explicitly permits redistribution.
7. Interpret clusters as statistical patterns, not causal or diagnostic conclusions.
8. Record dataset source, download date, license/terms, filters, units, year, and limitations.
9. Standard K-Means uses squared Euclidean distance. Do not mislabel it as Manhattan K-Means.
10. No LLM API calls are required in the deployed application.

## File ownership
- Research/Data agent: `docs/data_provenance.md`, `docs/feature_dictionary.md`
- Data engineering agent: `src/data.py`
- ML agent: `src/clustering.py`
- UI/visualization agent: `app.py`, `src/visualization.py`
- QA/integration agent: `tests/`, `docs/testing.md`, integration review
- Project lead/instruction manager: approves interface contracts and merges changes

## Definition of done
- App starts from a clean environment using `requirements.txt`.
- Input validation gives useful errors for invalid CSVs.
- Missing values and scaling are handled reproducibly.
- At least three values of k can be evaluated.
- WCSS, Silhouette Score, and Davies–Bouldin Index are reported where mathematically defined.
- PCA plot and interpretable cluster profiles are available.
- Results can be downloaded.
- Tests pass and the live deployment URL is smoke-tested.
- Report documents provenance, methodology, limitations, bias, and responsible use.
