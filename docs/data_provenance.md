# IHME Global Burden of Disease 2023 — Data Provenance

## Source record

- **Publisher:** Institute for Health Metrics and Evaluation (IHME), University of Washington
- **Study:** Global Burden of Disease Study 2023 Results
- **Official Results Tool:** <https://vizhub.healthdata.org/gbd-results/>
- **Source file supplied locally:** `data/raw/IHME-GBD_2023_DATA-34567faa-1.csv`
- **Download date:** Not recorded in the supplied files.
- **Terms:** Access, use and redistribution are governed by the IHME Results Tool terms applicable to the downloader. Review those terms before sharing or redistributing data; this project does not assume public redistribution rights.

The supplied `data/raw/citation.txt` gives the following citation:

> Global Burden of Disease Collaborative Network. Global Burden of Disease Study 2023 (GBD 2023) Results. Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2024. Available from https://vizhub.healthdata.org/gbd-results/.

## Supplied extract verification

The local CSV was inspected and transformed through the project ingestion pipeline. It contains 3,060 long-format records, 204 distinct locations, 15 causes, and the year 2023. The supplied records have one value each for measure, metric, age, sex and population group:

- Measure: DALYs (Disability-Adjusted Life Years)
- Metric: Rate
- Age: Age-standardized
- Sex: Both
- Year: 2023
- Population group: All Population

The transformation yields 204 country/territory rows and 15 cause features. All 204 rows passed the default 30% missingness rule, with no imputations in the verified extract. This is a local verification result, not a model-quality or epidemiological conclusion.

The raw and processed data directories are ignored by Git. The supplied file is therefore not included in the repository or a cloud deployment. The app can use it locally when present, or accept an authorized upload.

## Reproducible query dimensions

For a comparable extract, use IHME GBD Results Tool settings matching the following:

| Dimension | Selection |
|---|---|
| Context | Causes |
| Measure | DALYs |
| Metric | Rate |
| Age | Age-standardized |
| Sex | Both |
| Year | Selected explicitly in the app; the supplied extract is 2023 |
| Location | Country and territory observations; exclude global/regional aggregates |
| Cause | The 15 cause categories listed in the [feature dictionary](feature_dictionary.md) |

Rates are used to compare burden relative to population size; age-standardization reduces differences attributable to national age structures. It does not remove all differences in measurement, estimation or source-data completeness.

## Ingestion and transformation

The GBD upload path expects `location_name`, `cause_name`, and `val`. If present, `measure_name`, `metric_name`, `sex_name`, and `age_name` are filtered to DALYs, Rate, Both and Age-standardized respectively. When `year` is present, the user selects one year in the dashboard. The source must have at least three resulting country/territory observations and two cause features.

Known global, regional, super-regional and WHO aggregate location names are excluded by `src/gbd_transform.py`. Each `(location_name, cause_name)` pair must be unique after filtering; duplicate estimates are rejected rather than silently aggregated. `val` is converted to numeric, and invalid values are treated as missing for the transparent preprocessing stage.

The transformation pivots `location_name` into the `country` identifier and `cause_name` into feature columns. Data then passes through country identifier validation, row-level missingness auditing, median imputation of retained observations and standard scaling.

## Limitations and responsible use

- GBD indicators are modeled estimates and may include uncertainty not represented in the point-estimate clustering matrix.
- The download date and account-specific terms were not included with the supplied file.
- Filtering known aggregates by name does not assert that every possible custom or newly named location is a country; inspect uploaded sources and output locations before analysis.
- Country-level estimates mask internal geographic, socioeconomic and demographic differences.
- Cluster membership describes patterns in the selected features and must not be interpreted as causality, diagnosis, country ranking or policy-effectiveness evidence.
