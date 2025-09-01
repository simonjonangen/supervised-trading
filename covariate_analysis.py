"""
Redundancy analysis for merged.csv

What it does
------------
1) Loads data and selects numeric features.
2) Flags constant / near-constant features.
3) Computes Pearson & Spearman correlations.
4) Clusters features by correlation and suggests keeping one per cluster
   (preferring higher mutual information with the target `label` if available,
   otherwise higher variance).
5) Computes VIF to flag multicollinearity.
6) Runs PCA to show effective dimensionality.
7) Writes a 'redundancy_report.csv' with suggested drops and reasons.

Requirements
------------
pip install pandas numpy scipy scikit-learn statsmodels
"""

from __future__ import annotations

import os
import math
import json
import warnings
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 200)

# ------------------------------ Config ---------------------------------

CSV_PATH = "data/merged.csv"

TIMESTAMP_COL = "timestamp"
LABEL_COL = "label"

NEAR_ZERO_VAR_THRESHOLD = 1e-6
HIGH_CORR_THRESHOLD = 0.95
SPEARMAN_TOO = True

VIF_THRESHOLD = 10.0

RANDOM_STATE = 42
REPORT_CSV = "data/redundancy_report.csv"

# -------------------------- Helper structures --------------------------

@dataclass
class FeatureStats:
    name: str
    variance: float
    near_constant: bool
    vif: Optional[float]
    mi_with_label: Optional[float]
    cluster_id: Optional[int]
    representative: bool
    representative_of: Optional[str]
    drop_reason: Optional[str]

# ---------------------------- Load data --------------------------------

df = pd.read_csv(CSV_PATH)


if TIMESTAMP_COL in df.columns:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        except Exception:
            pass

exclude_cols = {TIMESTAMP_COL}
if LABEL_COL in df.columns:
    exclude_cols.add(LABEL_COL)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [c for c in num_cols if c not in exclude_cols]

if not features:
    raise RuntimeError("No numeric feature columns found to analyze.")

y = None
y_is_classification = None
if LABEL_COL in df.columns:
    y = df[LABEL_COL]

    if pd.api.types.is_bool_dtype(y) or (pd.api.types.is_integer_dtype(y) and y.nunique(dropna=True) <= max(20, int(0.05*len(y)))):
        y_is_classification = True
    elif pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) < 10:
        y_is_classification = True
    else:
        y_is_classification = False

df_features = df[features].copy()
row_all_nan = df_features.isna().all(axis=1)
df_features = df_features.loc[~row_all_nan].reset_index(drop=True)
if y is not None:
    y = y.loc[~row_all_nan].reset_index(drop=True)

# Simple imputation: fill remaining NaNs with column medians
df_features = df_features.apply(lambda s: s.fillna(s.median()))

# --------------------- Constant / near-constant ------------------------

variances = df_features.var(ddof=0)
near_constant_mask = variances < NEAR_ZERO_VAR_THRESHOLD

# -------------------------- Correlations --------------------------------

# Pearson
pearson_corr = df_features.corr(method="pearson").fillna(0.0)
# Spearman (optional)
spearman_corr = df_features.corr(method="spearman").fillna(0.0) if SPEARMAN_TOO else None

# For clustering distance: use max of Pearson/Spearman distances when both used.
def corr_to_dist(corr_mat: pd.DataFrame) -> pd.DataFrame:
    # distance = 1 - |corr|
    return 1.0 - corr_mat.abs()

if SPEARMAN_TOO:
    d1 = corr_to_dist(pearson_corr)
    d2 = corr_to_dist(spearman_corr)
    # combine conservatively (take min similarity -> max distance)
    dist = pd.DataFrame(np.maximum(d1.values, d2.values), index=features, columns=features)
else:
    dist = corr_to_dist(pearson_corr)

np.fill_diagonal(dist.values, 0.0)
condensed = squareform(dist.values, checks=False)


Z = linkage(condensed, method="average")
# Convert correlation threshold to distance threshold:
# keep together if |corr| >= HIGH_CORR_THRESHOLD -> distance <= 1 - HIGH_CORR_THRESHOLD
cluster_thresh = 1.0 - HIGH_CORR_THRESHOLD
clusters = fcluster(Z, t=cluster_thresh, criterion="distance")

cluster_map: Dict[int, List[str]] = {}
for col, cid in zip(features, clusters):
    cluster_map.setdefault(cid, []).append(col)

# ---------------------- Mutual Information (optional) -------------------

mi_scores: Dict[str, Optional[float]] = {f: None for f in features}
if y is not None and pd.api.types.is_numeric_dtype(y):
    X = df_features.values
    if y_is_classification:
        y_input = y.astype(int).values
        mi_vals = mutual_info_classif(X, y_input, random_state=RANDOM_STATE, discrete_features=False)
    else:
        y_input = y.values
        mi_vals = mutual_info_regression(X, y_input, random_state=RANDOM_STATE, discrete_features=False)
    mi_scores = {f: float(m) for f, m in zip(features, mi_vals)}

# --------------------- Cluster representatives --------------------------

# Pick a representative for each cluster:
#  1) if MI is available: highest MI with label
#  2) else: highest variance
rep_for: Dict[str, str] = {}       # feature -> representative of its cluster
rep_is: Dict[str, bool] = {f: False for f in features}

for cid, cols in cluster_map.items():
    if len(cols) == 1:
        rep = cols[0]
    else:
        if any(mi_scores[c] is not None for c in cols):
            rep = max(cols, key=lambda c: (mi_scores[c] if mi_scores[c] is not None else -np.inf, variances[c]))
        else:
            rep = max(cols, key=lambda c: variances[c])
    rep_is[rep] = True
    for c in cols:
        rep_for[c] = rep

# ----------------------------- VIF --------------------------------------

# Compute VIF on standardized features and only for non-near-constant columns
vif_values: Dict[str, Optional[float]] = {f: None for f in features}
keep_for_vif = [f for f in features if not near_constant_mask.get(f, False)]
if len(keep_for_vif) >= 2:
    X_std = pd.DataFrame(StandardScaler().fit_transform(df_features[keep_for_vif]), columns=keep_for_vif)
    for i, col in enumerate(keep_for_vif):
        try:
            vif_values[col] = float(variance_inflation_factor(X_std.values, i))
        except Exception:
            vif_values[col] = None

# ------------------------- PCA (dimensionality) -------------------------

pca_info = {}
try:
    X_sc = StandardScaler().fit_transform(df_features[features])
    pca = PCA().fit(X_sc)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k90 = int(np.searchsorted(cumvar, 0.90) + 1)
    k95 = int(np.searchsorted(cumvar, 0.95) + 1)
    pca_info = {
        "n_features": len(features),
        "k90_variance": k90,
        "k95_variance": k95,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist()
    }
except Exception as e:
    pca_info = {"error": str(e)}

# --------------------- Build redundancy decisions -----------------------

rows: List[FeatureStats] = []

for f in features:
    drop_reason = None
    representative = rep_is[f]
    cluster_id = int(rep_for[f] == f and clusters[features.index(f)] or clusters[features.index(rep_for[f])])

    # near-constant dominates
    if near_constant_mask.get(f, False):
        drop_reason = "near-constant variance"
    else:
        # if not representative and correlated beyond threshold, mark redundant
        if not representative and (abs(pearson_corr.loc[f, rep_for[f]]) >= HIGH_CORR_THRESHOLD or
                                   (SPEARMAN_TOO and abs(spearman_corr.loc[f, rep_for[f]]) >= HIGH_CORR_THRESHOLD)):
            drop_reason = f"highly correlated with {rep_for[f]} (>= {HIGH_CORR_THRESHOLD})"

    # VIF suggestion (only if still not dropped and VIF high)
    vif_val = vif_values.get(f, None)
    if drop_reason is None and vif_val is not None and math.isfinite(vif_val) and vif_val >= VIF_THRESHOLD:
        drop_reason = f"high VIF ({vif_val:.2f} ≥ {VIF_THRESHOLD})"

    rows.append(FeatureStats(
        name=f,
        variance=float(variances[f]),
        near_constant=bool(near_constant_mask.get(f, False)),
        vif=vif_val if (vif_val is None or math.isfinite(vif_val)) else None,
        mi_with_label=mi_scores.get(f, None),
        cluster_id=int(cluster_id),
        representative=representative,
        representative_of=(None if representative else rep_for[f]),
        drop_reason=drop_reason
    ))

report_df = pd.DataFrame([asdict(r) for r in rows]).sort_values(
    by=["drop_reason"].copy(), ascending=[True]
)

# Put suggested_keep/suggested_drop columns up front
report_df.insert(1, "suggested_keep", report_df["drop_reason"].isna())
report_df.insert(2, "suggested_drop", ~report_df["drop_reason"].isna())

# Save report
report_df.to_csv(REPORT_CSV, index=False)

# ---------------------------- Console output ----------------------------

print("\n=== Summary ===")
print(f"Input file: {CSV_PATH}")
print(f"Features analyzed: {len(features)}")
print(f"Near-constant features: {int(report_df['near_constant'].sum())}")
print(f"Suggested drops (any reason): {int(report_df['suggested_drop'].sum())}")

if pca_info and "error" not in pca_info:
    print(f"\nPCA dimensionality: k90={pca_info['k90_variance']}, k95={pca_info['k95_variance']} out of {pca_info['n_features']}")
else:
    print("\nPCA skipped/failed:", pca_info.get("error", ""))

print("\nTop 15 rows of redundancy report:")
print(report_df.head(15).to_string(index=False))

print(f"\nReport written to: {os.path.abspath(REPORT_CSV)}")

# ------------------------- Optional: quick tips -------------------------

print("\nHow to use:")
print("- Drop columns where suggested_drop == True to reduce redundancy.")
print("- If you prefer keeping a different member of a correlated cluster, pick one with higher MI (if label exists) or variance.")
print("- Re-check VIF after dropping highly correlated features; VIF often falls once redundant columns are removed.")

# ------------------------ Save reduced dataset --------------------------

# Columns to keep: timestamp + label (if present) + suggested_keep features
keep_features = report_df.loc[report_df["suggested_keep"], "name"].tolist()

cols_to_save = []
if TIMESTAMP_COL in df.columns:
    cols_to_save.append(TIMESTAMP_COL)
cols_to_save.extend(keep_features)
if LABEL_COL in df.columns:
    cols_to_save.append(LABEL_COL)

reduced_df = df[cols_to_save].copy()
reduced_path = "data/merged_reduced.csv"
reduced_df.to_csv(reduced_path, index=False)

print(f"\nReduced dataset saved to: {os.path.abspath(reduced_path)}")
print(f"Kept {len(keep_features)} features (plus timestamp/label if present).")
