from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def validate_dataset(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Validate that the DataFrame adheres to the Country Disease-Burden dataset contract.

    Contract rules:
    - Must be non-empty and have at least 3 country observations.
    - Must contain a usable country identifier column.
    - Country identifiers must be non-null and non-blank.
    - Duplicate country identifiers are rejected (must be 1 observation per country).
    - Must contain at least two usable numeric candidate feature columns.
    """
    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")
    if len(df) < 3:
        raise ValueError("The dataset must contain at least three country rows.")
    if len(df.columns) < 2:
        raise ValueError("The CSV must contain a country identifier and numeric features.")

    country_candidates = [
        c for c in df.columns
        if str(c).strip().lower() in {
            "country", "country_name", "country name", "location", "location_name",
            "entity", "iso3", "iso_code", "country_code", "country code",
            "nation", "state", "territory", "economy"
        }
    ]
    if not country_candidates:
        raise ValueError(
            "No country identifier column found. The dataset must contain a column named "
            "'country', 'location', 'entity', 'iso3', or similar."
        )
    country_col = country_candidates[0]

    if country_col not in df.columns:
        raise ValueError(f"Country identifier column '{country_col}' not found in dataset.")

    # Validate non-null and non-blank country identifiers
    country_series = df[country_col]
    country_str = country_series.astype(str).str.strip()
    is_blank = (
        country_series.isna()
        | (country_str == "")
        | (country_str.str.lower() == "nan")
        | (country_str.str.lower() == "none")
        | (country_str.str.lower() == "null")
    )
    if is_blank.any():
        blank_count = int(is_blank.sum())
        blank_indices = df.index[is_blank].tolist()[:5]
        raise ValueError(
            f"Country identifier column '{country_col}' contains {blank_count} null or blank value(s) "
            f"(e.g. at row indices {blank_indices}). Every observation must have a valid country name or code."
        )

    # Validate no duplicate country records
    duplicates = country_str[country_str.duplicated(keep=False)].unique().tolist()
    if duplicates:
        preview = ", ".join(repr(d) for d in duplicates[:5])
        suffix = f" (and {len(duplicates) - 5} more)" if len(duplicates) > 5 else ""
        raise ValueError(
            f"Duplicate country identifiers detected: {preview}{suffix}. "
            "The dataset contract requires exactly one comparable observation per country. "
            "Duplicate rows typically indicate multiple years, age groups, sexes, or disease measures "
            "that must be filtered, averaged, or harmonized before clustering."
        )

    # Detect usable numeric feature columns
    numeric_cols = []
    for col in df.columns:
        if col == country_col:
            continue
        converted = pd.to_numeric(df[col].replace([np.inf, -np.inf], np.nan), errors="coerce")
        if converted.notna().sum() >= max(3, int(0.5 * len(df))):
            numeric_cols.append(col)

    if len(numeric_cols) < 2:
        raise ValueError(
            f"At least two usable numeric feature columns are required, but found {len(numeric_cols)}: {numeric_cols}."
        )

    return country_col, numeric_cols


def prepare_features(
    df: pd.DataFrame,
    features: list[str],
    max_missing_fraction: float = 0.30,
    country_col: str | None = None,
) -> dict:
    """Preprocess selected features with configurable row-level missingness filtering.

    Missingness Policy:
    - max_missing_fraction defines the maximum allowed proportion of missing features per row (default: 0.30).
    - Rows with missing_fraction <= max_missing_fraction are retained.
    - Rows with missing_fraction > max_missing_fraction are excluded.
    - Excluded rows are recorded in an audit DataFrame with reason codes.
    - Median imputation is applied strictly to remaining missing cells in retained rows.
    """
    if not (0.0 <= max_missing_fraction <= 1.0):
        raise ValueError(
            f"max_missing_fraction must be between 0.0 and 1.0, got {max_missing_fraction}."
        )
    if len(features) < 2:
        raise ValueError("Select at least two numeric features.")
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Feature columns not found: {missing_cols}")

    if country_col is None:
        country_candidates = [
            c for c in df.columns
            if str(c).strip().lower() in {
                "country", "country_name", "country name", "location", "location_name",
                "entity", "iso3", "iso_code", "country_code", "country code",
                "nation", "state", "territory", "economy"
            }
        ]
        country_col = country_candidates[0] if country_candidates else str(df.columns[0])

    # Convert to numeric, explicitly coercing infinities to NaN
    numeric = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    n_features = len(features)

    # Calculate row-level missingness
    missing_counts = numeric.isna().sum(axis=1)
    missing_fractions = missing_counts / n_features

    # Evaluate retention rule: exactly at or below threshold is retained
    eligible_mask = missing_fractions <= max_missing_fraction
    retained_numeric = numeric.loc[eligible_mask]
    retained_indices = retained_numeric.index.tolist()

    # Build exclusion audit records
    excluded_mask = ~eligible_mask
    excluded_indices = df.index[excluded_mask].tolist()
    exclusion_records = []
    for idx in excluded_indices:
        c_name = str(df.loc[idx, country_col]) if country_col in df.columns else str(idx)
        m_count = int(missing_counts.loc[idx])
        m_frac = float(missing_fractions.loc[idx])
        reason = "ALL_FEATURES_MISSING" if m_count == n_features else "EXCEEDS_MISSING_THRESHOLD"
        exclusion_records.append({
            "country": c_name,
            "row_index": idx,
            "missing_count": m_count,
            "total_features": n_features,
            "missing_fraction": round(m_frac, 4),
            "missing_percentage": f"{m_frac * 100:.1f}%",
            "reason": reason,
        })

    exclusions_df = pd.DataFrame(
        exclusion_records,
        columns=[
            "country", "row_index", "missing_count", "total_features",
            "missing_fraction", "missing_percentage", "reason"
        ]
    )

    if len(retained_numeric) < 3:
        pct_label = int(round(max_missing_fraction * 100))
        raise ValueError(
            f"Fewer than three usable country rows remain after applying the {pct_label}% "
            f"missingness threshold (retained {len(retained_numeric)} of {len(df)} rows)."
        )

    # Check that no retained column is completely NaN
    all_nan_cols = [c for c in features if retained_numeric[c].isna().all()]
    if all_nan_cols:
        raise ValueError(f"Feature column(s) have no observed values among retained rows: {all_nan_cols}")

    # Impute missing cells for eligible rows only
    imputed_cells = int(retained_numeric.isna().sum().sum())
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(retained_numeric)

    if not np.isfinite(imputed).all():
        raise ValueError("Features contain non-finite values after imputation.")

    stds = np.nanstd(imputed, axis=0)
    if np.any(stds == 0):
        constant = [features[i] for i, sd in enumerate(stds) if sd == 0]
        raise ValueError(f"Remove constant feature columns: {constant}")

    scaler = StandardScaler()
    X = scaler.fit_transform(imputed)

    return {
        "X": X,
        "row_indices": retained_indices,
        "total_rows": len(df),
        "retained_rows": len(retained_indices),
        "excluded_rows": len(excluded_indices),
        "exclusions_df": exclusions_df,
        "imputed_cells": imputed_cells,
        "imputer": imputer,
        "scaler": scaler,
        "features": features,
        "max_missing_fraction": max_missing_fraction,
        "country_col": country_col,
    }
