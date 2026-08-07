"""Build the retrospective admission-level ICD-coded septic-shock dataset.

Features summarize available measurements in the first 120 hours after admission.
``septic_shock`` is discharge diagnosis ICD-9 78552. This is retrospective
identification, not prediction after a 120-hour landmark.
"""
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
import duckdb
import warnings

warnings.filterwarnings('ignore')
SCRIPT_DIR = Path(__file__).resolve().parent
con = duckdb.connect("mimic3.db")

sql_bg = (SCRIPT_DIR / "hmm-data.sql").read_text(encoding="utf-8-sig")
df = con.execute(sql_bg).fetch_df()
con.close()
# DuckDB may preserve uppercase names for identifiers originating in MIMIC.
# Normalize once so SQL-source casing cannot break the shared construction code.
df.columns = [str(column).lower() for column in df.columns]
if df.columns.duplicated().any():
    duplicates = df.columns[df.columns.duplicated()].tolist()
    raise ValueError(f"Column-name normalization created duplicates: {duplicates}")

keep_cols = [
    "subject_id",
    "hadm_id",
    'septic_shock',
    "heart_rate_6h",
    "resp_rate_6h",
    "temperature_6h",
    "sbp_6h",
    "wbc_24h",
    "platelets_24h",
    "creatinine_24h",
    "inr_24h",
    "lactate_24h",
    "bilirubin_24h",
]

missing = [c for c in keep_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in df: {missing}\nAvailable columns: {list(df.columns)}")

df = df.loc[:, keep_cols].copy()

# SQL retains the latest adult admission per patient. Preserve identifiers for
# auditing only; downstream models use an explicit feature list that excludes them.
if df["subject_id"].duplicated().any():
    raise ValueError("Expected one retained admission per patient, but subject_id is duplicated.")
if df["hadm_id"].duplicated().any():
    raise ValueError("Expected unique retained admissions, but hadm_id is duplicated.")

import numpy as np
import pandas as pd
import ast

feature_cols = [
    "heart_rate_6h",
    "resp_rate_6h",
    "temperature_6h",
    "sbp_6h",
    "wbc_24h",
    "platelets_24h",
    "creatinine_24h",
    "inr_24h",
    "lactate_24h",
    "bilirubin_24h",
]

MISSING_STRINGS = {"", "-", "na", "n/a", "nan", "none", "null"}

def to_float_or_nan(v):
    if v is None:
        return np.nan
    if isinstance(v, float) and np.isnan(v):
        return np.nan
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in MISSING_STRINGS:
            return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan

def parse_string_vector(s: str):
    s = s.strip()
    if s == "" or s.lower() in MISSING_STRINGS:
        return None

    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            return ast.literal_eval(s)
        except Exception:
            pass

    if "," in s:
        return [tok.strip() for tok in s.split(",")]

    return s

def cell_vector(x, expected_bins):
    """Parse one SQL array and verify its expected longitudinal bin count."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.full(expected_bins, np.nan, dtype=float)
    if isinstance(x, str):
        parsed = parse_string_vector(x)
        if parsed is None:
            return np.full(expected_bins, np.nan, dtype=float)
        x = parsed
    if not isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        raise ValueError(f"Expected a longitudinal vector, received {type(x).__name__}.")
    arr = np.asarray([to_float_or_nan(v) for v in x], dtype=float)
    if len(arr) != expected_bins:
        raise ValueError(f"Expected {expected_bins} bins, received {len(arr)}.")
    return arr


def summarize_longitudinal_cell(x, expected_bins, bin_hours):
    """Return mean, standard deviation, last value, and trend per hour."""
    arr = cell_vector(x, expected_bins)
    observed = np.flatnonzero(~np.isnan(arr))
    if len(observed) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "last": np.nan,
            "trend_per_hour": np.nan,
        }

    values = arr[observed]
    times = observed.astype(float) * bin_hours
    trend = np.nan
    if len(observed) >= 2 and np.ptp(times) > 0:
        trend = float(np.polyfit(times, values, deg=1)[0])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "last": float(values[-1]),
        "trend_per_hour": trend,
    }

BIN_SPECS = {
    "heart_rate_6h": (20, 6),
    "resp_rate_6h": (20, 6),
    "temperature_6h": (20, 6),
    "sbp_6h": (20, 6),
    "wbc_24h": (5, 24),
    "platelets_24h": (5, 24),
    "creatinine_24h": (5, 24),
    "inr_24h": (5, 24),
    "lactate_24h": (5, 24),
    "bilirubin_24h": (5, 24),
}

summary_frames = []
for feature_name in feature_cols:
    expected_bins, bin_hours = BIN_SPECS[feature_name]
    summaries = df[feature_name].apply(
        lambda x: summarize_longitudinal_cell(x, expected_bins, bin_hours)
    )
    summary = pd.DataFrame(summaries.tolist(), index=df.index)
    # Preserve the legacy base name as the longitudinal mean; suffix all other summaries.
    summary = summary.rename(columns={
        stat: feature_name if stat == "mean" else f"{feature_name}_{stat}"
        for stat in summary.columns
    })
    summary_frames.append(summary)

identifier_and_target = df.drop(columns=feature_cols)
df = pd.concat([identifier_and_target, *summary_frames], axis=1)
# Preserve missing values in the exported CSV. Imputation must be fitted using
# training data only in the downstream analysis (and separately within each
# cross-validation training fold), then applied to validation and test data.

output_path = SCRIPT_DIR / "septicshock.csv"
df.to_csv(output_path, index=False)
print(f"Wrote {len(df):,} rows and {df.shape[1]} columns to {output_path}")












