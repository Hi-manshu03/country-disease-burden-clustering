"""GBD Long-to-Wide Transformation Pipeline.

Transforms raw long-format CSV datasets downloaded from the IHME GBD Results Tool
into the country-level wide-format matrix required by Project 28.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Known GBD global and super-regional aggregate locations to filter out
# when retaining country and territory observations only.
GBD_REGIONAL_AGGREGATES = {
    "global",
    "world",
    "high-income",
    "high-income north america",
    "high-income asia pacific",
    "western europe",
    "australasia",
    "southern latin america",
    "central europe, eastern europe, and central asia",
    "central europe",
    "eastern europe",
    "central asia",
    "sub-saharan africa",
    "central sub-saharan africa",
    "eastern sub-saharan africa",
    "southern sub-saharan africa",
    "western sub-saharan africa",
    "north africa and middle east",
    "south asia",
    "southeast asia, east asia, and oceania",
    "east asia",
    "southeast asia",
    "oceania",
    "latin america and caribbean",
    "andean latin america",
    "caribbean",
    "central latin america",
    "tropical latin america",
    "world bank high income",
    "world bank low income",
    "world bank lower middle income",
    "world bank upper middle income",
    "who region",
    "african region",
    "region of the americas",
    "south-east asia region",
    "european region",
    "eastern mediterranean region",
    "western pacific region",
}

DEFAULT_GBD_CAUSES = [
    "Cardiovascular diseases",
    "Neoplasms",
    "Chronic respiratory diseases",
    "Digestive diseases",
    "Neurological disorders",
    "Mental disorders",
    "Substance use disorders",
    "Diabetes and kidney diseases",
    "Respiratory infections and tuberculosis",
    "Enteric infections",
    "Neglected tropical diseases and malaria",
    "Maternal and neonatal disorders",
    "Nutritional deficiencies",
    "Musculoskeletal disorders",
    "Self-harm and interpersonal violence",
]


def transform_gbd_long_to_wide(
    long_df: pd.DataFrame,
    selected_causes: list[str] | None = None,
    filter_aggregates: bool = True,
    target_year: int | None = None,
) -> pd.DataFrame:
    """Transform an official IHME GBD long-format DataFrame into a country-by-feature matrix.

    Parameters
    ----------
    long_df : pd.DataFrame
        Raw DataFrame as imported from GBD Results Tool CSV export.
    selected_causes : list[str] or None, optional
        Subset of causes to extract as columns. If None, uses DEFAULT_GBD_CAUSES
        or all available causes matching the filter.
    filter_aggregates : bool, default True
        Whether to remove known global and super-regional aggregate entities.
    target_year : int or None, optional
        Filter to a specific year if the source file contains multiple years.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with 'country' as identifier and one column per disease feature.

    Raises
    ------
    ValueError
        If required schema columns are missing, no records match filters,
        or duplicate (location, cause) observations are detected.
    """
    if long_df.empty:
        raise ValueError("The provided GBD DataFrame is empty.")

    # 1. Schema Validation
    required_cols = {"location_name", "cause_name", "val"}
    missing_cols = required_cols - set(long_df.columns)
    if missing_cols:
        raise ValueError(f"GBD long-format file is missing required columns: {sorted(missing_cols)}")

    df = long_df.copy()

    # 2. Filter by Measure and Metric if columns present
    if "measure_name" in df.columns:
        # Match 'DALYs (Disability-Adjusted Life Years)' or 'DALYs'
        daly_mask = df["measure_name"].astype(str).str.contains("DALY", case=False, na=False)
        if daly_mask.any():
            df = df.loc[daly_mask]

    if "metric_name" in df.columns:
        # Target 'Rate'
        rate_mask = df["metric_name"].astype(str).str.strip().str.lower() == "rate"
        if rate_mask.any():
            df = df.loc[rate_mask]

    if "sex_name" in df.columns:
        # Target 'Both'
        sex_mask = df["sex_name"].astype(str).str.strip().str.lower() == "both"
        if sex_mask.any():
            df = df.loc[sex_mask]

    if "age_name" in df.columns:
        # Target 'Age-standardized'
        age_mask = df["age_name"].astype(str).str.contains("age-standardized", case=False, na=False)
        if age_mask.any():
            df = df.loc[age_mask]

    # 3. Filter Target Year if specified
    if target_year is not None and "year" in df.columns:
        df = df.loc[df["year"] == target_year]
    elif "year" in df.columns and df["year"].nunique() > 1:
        # If multiple years present and target_year not specified, take the maximum year
        max_year = int(df["year"].max())
        df = df.loc[df["year"] == max_year]

    if df.empty:
        raise ValueError("No records remain after applying standard GBD dimension filters.")

    # 4. Filter Geographic Aggregates
    if filter_aggregates:
        is_agg = df["location_name"].astype(str).str.strip().str.lower().isin(GBD_REGIONAL_AGGREGATES)
        df = df.loc[~is_agg]

    # 5. Filter Selected Causes
    if selected_causes is not None:
        causes_to_keep = set(selected_causes)
        df = df.loc[df["cause_name"].isin(causes_to_keep)]
    else:
        # If default causes exist in df, prioritize them
        available_default = [c for c in DEFAULT_GBD_CAUSES if c in df["cause_name"].values]
        if len(available_default) >= 2:
            df = df.loc[df["cause_name"].isin(available_default)]

    if df.empty:
        raise ValueError("No records match the requested cause categories.")

    # 6. Verify Uniqueness of (location_name, cause_name)
    # The GBD contract requires exactly one comparable estimate per country-cause.
    dup_mask = df.duplicated(subset=["location_name", "cause_name"], keep=False)
    if dup_mask.any():
        sample_dups = df.loc[dup_mask, ["location_name", "cause_name"]].drop_duplicates().head(5)
        dup_list = [f"{r.location_name} - {r.cause_name}" for _, r in sample_dups.iterrows()]
        raise ValueError(
            f"Duplicate (location, cause) observations detected: {dup_list}. "
            "Input data contains multiple estimates for the same country and disease feature. "
            "Check for unharmonized age groups, sexes, or multi-year slices."
        )

    # 7. Convert val to numeric, coercing invalid values to NaN
    df["val"] = pd.to_numeric(df["val"].replace([np.inf, -np.inf], np.nan), errors="coerce")

    # 8. Pivot Long to Wide
    wide = df.pivot(index="location_name", columns="cause_name", values="val")
    wide = wide.reset_index().rename(columns={"location_name": "country"})

    # Clean columns index name
    wide.columns.name = None

    if len(wide) < 3:
        raise ValueError(f"Fewer than three countries remain in transformed dataset (got {len(wide)}).")

    feature_cols = [c for c in wide.columns if c != "country"]
    if len(feature_cols) < 2:
        raise ValueError(f"Fewer than two disease features present in transformed dataset (got {len(feature_cols)}).")

    return wide

