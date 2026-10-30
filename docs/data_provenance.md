# Official IHME / GBD Data Provenance Record

## 1. Study and Source Metadata
- **Dataset Title:** Global Burden of Disease (GBD) Study — Cause-Specific Disability-Adjusted Life Years (DALYs)
- **Publisher / Source:** Institute for Health Metrics and Evaluation (IHME), University of Washington
- **Official Tool URL:** [https://vizhub.healthdata.org/gbd-results/](https://vizhub.healthdata.org/gbd-results/)
- **Catalog Portal:** Global Health Data Exchange (GHDx) ([https://ghdx.healthdata.org/](https://ghdx.healthdata.org/))
- **Study / Release Version:** GBD 2021 / GBD 2023 (latest finalized release available in Results Tool)
- **Access Protocol:** Interactive query via IHME VizHub GBD Results Tool (requires free user registration)
- **License / Terms of Use:** IHME Non-Commercial User Agreement (Educational and research use permitted; commercial redistribution prohibited without explicit written permission)
- **Citation:** 
  > GBD 2021 Diseases and Injuries Collaborators. Global incidence, prevalence, years lived with disability (YLDs), disability-adjusted life-years (DALYs), and healthy life expectancy (HALE) for 371 diseases and injuries in 204 countries and territories and 811 subnational locations, 1990–2021: a systematic analysis for the Global Burden of Disease Study 2021. *The Lancet*, 2024; 403(10440): 2133–2161. DOI: 10.1016/S0140-6768(24)00757-8.

---

## 2. Target Dataset Query Specifications (for Project Owner Download)

To obtain the exact dataset needed for Project 28 clustering, configure the filters in the **GBD Results Tool** ([https://vizhub.healthdata.org/gbd-results/](https://vizhub.healthdata.org/gbd-results/)) as follows:

| Parameter | Selection in Tool | Methodological Rationale |
| :--- | :--- | :--- |
| **Base** | Single / GBD Estimates | Standard point estimates with uncertainty intervals |
| **Context** | Causes | Direct disease and injury burden (not risk factors or impairments) |
| **Measure** | DALYs (Disability-Adjusted Life Years) | Quantifies total health loss by summing premature mortality (YLL) and morbidity (YLD) |
| **Metric** | Rate | Expresses burden per 100,000 population, enabling meaningful cross-national comparison |
| **Age** | Age-standardized | Removes the confounding effect of differing population age structures between aging and youthful nations |
| **Sex** | Both | Captures aggregate population disease burden without sex disaggregation |
| **Year** | Latest available (2021 or 2023) | Provides the most recent comparable snapshot of global health patterns |
| **Location** | Select All Countries and Territories | Restrict to national units (204 GBD countries and territories); omit global and regional super-aggregates |
| **Cause** | 15 Selected Level 2 Causes (see Section 3) | Mutually exclusive, non-overlapping Level 2 categories spanning NCDs, communicable diseases, and injuries |

---

## 3. Selected Cause Categories (Level 2 Hierarchy)

To prevent multicollinearity, double-counting, and parent-child redundancy, the selection strictly utilizes **Level 2** disease and injury categories from the official GBD hierarchy:

1. `Cardiovascular diseases`
2. `Neoplasms`
3. `Chronic respiratory diseases`
4. `Digestive diseases`
5. `Neurological disorders`
6. `Mental disorders`
7. `Substance use disorders`
8. `Diabetes and kidney diseases`
9. `Respiratory infections and tuberculosis`
10. `Enteric infections`
11. `Neglected tropical diseases and malaria`
12. `Maternal and neonatal disorders`
13. `Nutritional deficiencies`
14. `Musculoskeletal disorders`
15. `Self-harm and interpersonal violence`

---

## 4. Source File Schema & Transformation Protocol

### Expected Raw Long-Format Schema
When downloaded from the GBD Results Tool, the CSV has the following standardized columns:
- `measure_id`, `measure_name` (e.g. `DALYs (Disability-Adjusted Life Years)`)
- `location_id`, `location_name` (e.g. `France`, `Japan`, `Nigeria`)
- `sex_id`, `sex_name` (e.g. `Both`)
- `age_id`, `age_name` (e.g. `Age-standardized`)
- `cause_id`, `cause_name` (e.g. `Cardiovascular diseases`)
- `metric_id`, `metric_name` (e.g. `Rate`)
- `year` (e.g. `2021` or `2023`)
- `val` (Point estimate: age-standardized DALYs per 100,000)
- `upper` (95% upper uncertainty interval)
- `lower` (95% lower uncertainty interval)

### Transformation Steps into Wide Clustering Contract
1. **Filter Verification:** Filter strictly for `measure_name == 'DALYs (Disability-Adjusted Life Years)'`, `metric_name == 'Rate'`, `sex_name == 'Both'`, `age_name == 'Age-standardized'`, and target `year`.
2. **Location Hygiene:** Exclude global, regional, and subnational aggregates (e.g. `Global`, `High-income North America`, `WHO Western Pacific Region`) to retain sovereign countries and recognized territories.
3. **Uniqueness Audit:** Verify that `(location_name, cause_name)` pairs are strictly unique. Reject duplicates without silent averaging.
4. **Reshaping:** Pivot the data so each row represents one `location_name` (labeled `country`) and each column represents a selected `cause_name` containing the `val` point estimate.
5. **Contract Validation:** Pass the wide DataFrame to `validate_dataset(df)` and `prepare_features(df, features)` to ensure row-level missingness auditing and standard scaling.

---

## 5. Storage and Data Hygiene Rules
- Original downloaded files must be stored in `project28_starter/data/raw/` (excluded from version control via `.gitignore`).
- Transformed wide-format datasets must be stored in `project28_starter/data/processed/`.
- No proprietary or restricted raw data will be committed to public Git repositories.

## Dataset citation

Global Burden of Disease Collaborative Network.
Global Burden of Disease Study 2023 (GBD 2023) Results.
Seattle, United States: Institute for Health Metrics and Evaluation (IHME), 2024.
Available from: https://vizhub.healthdata.org/gbd-results/
