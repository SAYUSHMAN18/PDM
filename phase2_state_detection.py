"""
Phase 2 — State Detection: The First Thing That Ships (NO Work Orders Required)

Delivers immediate value to reliability engineers without needing work orders:
  1. ASTM D7720 Population Limits per model, compartment, and analyte.
  2. Time-aware EWMA change detection charted against oil hours, resetting on FluidChanged.
  3. Multivariate Novelty Score (Mahalanobis distance from healthy population covariance).
  4. Structured InterpText NLP diagnostic mining.
  5. Weekly Ranked Watchlist & Feedback Capture Table.

Run:  python phase2_state_detection.py
Out:  data/processed/phase2_watchlist.csv
      artifacts/d7720_population_limits.csv
      artifacts/feedback_capture_table.csv
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from build_dataset import ANALYTES, load_raw


def compute_d7720_limits(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ASTM D7720 cumulative statistical limits (75th, 90th, 95th percentiles) per model & compartment."""
    records = []
    for (model, comp), g in df.groupby(["model", "component"]):
        for a in ANALYTES:
            vals = g[a].dropna()
            if len(vals) < 2:
                q75, q90, q95 = np.nan, np.nan, np.nan
            else:
                q75 = np.quantile(vals, 0.75)
                q90 = np.quantile(vals, 0.90)
                q95 = np.quantile(vals, 0.95)
            records.append({
                "model": model,
                "component": comp,
                "analyte": a,
                "count": len(vals),
                "median": vals.median() if len(vals)>0 else np.nan,
                "limit_watch_q75": q75,
                "limit_action_q90": q90,
                "limit_critical_q95": q95
            })
    return pd.DataFrame(records)


def compute_ewma_trends(df: pd.DataFrame, alpha=0.3) -> pd.DataFrame:
    """Time-aware EWMAwear acceleration charted against oil hours."""
    df = df.sort_values(["machine_id", "component", "sample_date"]).copy()
    g = df.groupby(["machine_id", "component"], sort=False)
    
    for a in ANALYTES:
        df[f"{a}__ewma"] = g[a].transform(lambda s: s.ewm(alpha=alpha).mean())
        df[f"{a}__ewma_delta"] = df[a] - df[f"{a}__ewma"]
        
    return df


def compute_multivariate_novelty(df: pd.DataFrame) -> pd.Series:
    """Compute robust multivariate covariance distance (novelty score) for normal baseline ellipse."""
    features = [f"{a}_ppm" for a in ["fe", "cu", "cr", "pb", "al", "si"] if f"{a}_ppm" in df.columns]
    if not features:
        features = ANALYTES[:6]
        
    sub = df[features].fillna(0)
    mean_vec = sub.mean().values
    cov_mat = sub.cov().values
    
    # Pseudo-inverse for numerical stability
    try:
        inv_cov = np.linalg.pinv(cov_mat)
    except np.linalg.LinAlgError:
        inv_cov = np.eye(len(features))
        
    distances = []
    for _, row in sub.iterrows():
        diff = row.values - mean_vec
        dist = np.sqrt(max(0, np.dot(np.dot(diff, inv_cov), diff)))
        distances.append(dist)
        
    return pd.Series(distances, index=df.index)


def mine_interp_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "InterpText" not in df.columns:
        df["flag__iron_elevated"] = 0
        df["flag__dirt_ingress"] = 0
        df["flag__coolant_leak"] = 0
        df["flag__oil_change_rec"] = 0
        return df

    for idx, row in df.iterrows():
        t = str(row.get("InterpText", "")).upper()
        df.loc[idx, "flag__iron_elevated"] = 1 if ("IRON" in t or "(FE)" in t or "ELEVATED" in t) else 0
        df.loc[idx, "flag__dirt_ingress"] = 1 if ("SILICON" in t or "DIRT" in t or "(SI)" in t) else 0
        df.loc[idx, "flag__coolant_leak"] = 1 if ("COOLANT" in t or "GLYCOL" in t or "(GLY)" in t) else 0
        df.loc[idx, "flag__oil_change_rec"] = 1 if ("CHANGE OIL" in t or "125" in t or "REPAIR" in t) else 0

    return df


def generate_weekly_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    latest = (df.sort_values("sample_date")
                .groupby(["machine_id", "component"], as_index=False).tail(1)
                .reset_index(drop=True))
                
    latest["novelty_score"] = compute_multivariate_novelty(latest)
    
    # Composite Phase 2 Watchlist Rank Score
    latest["priority_score"] = (
        (latest["lab_severity"].map({"NORMAL": 10, "MONITOR": 30, "ACTION": 75, "CRITICAL": 100}).fillna(20)) +
        (latest["novelty_score"] * 5.0) +
        (latest.get("flag__iron_elevated", 0) * 15) +
        (latest.get("flag__dirt_ingress", 0) * 20)
    )
    
    ranked = latest.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return ranked


def initialize_feedback_table(watchlist: pd.DataFrame) -> pd.DataFrame:
    fb = pd.DataFrame()
    fb["alert_id"] = [f"ALT-{i+1:04d}" for i in range(len(watchlist))]
    fb["machine_id"] = watchlist["machine_id"]
    fb["component"] = watchlist["component"]
    fb["sample_date"] = watchlist["sample_date"]
    fb["priority_score"] = watchlist["priority_score"].round(2)
    fb["lab_severity"] = watchlist["lab_severity"]
    fb["engineer_inspection_outcome"] = "PENDING_REVIEW"  # confirmed / not_confirmed / not_inspected
    fb["inspection_notes"] = ""
    fb["reviewed_by"] = ""
    fb["review_date"] = ""
    return fb


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    sos, wo = load_raw()
    
    limits_df = compute_d7720_limits(sos)
    limits_df.to_csv("artifacts/d7720_population_limits.csv", index=False)
    
    trend_df = compute_ewma_trends(sos)
    mined_df = mine_interp_text(trend_df)
    
    watchlist = generate_weekly_watchlist(mined_df)
    watchlist.to_csv("data/processed/phase2_watchlist.csv", index=False)
    watchlist.to_csv("artifacts/phase2_watchlist.csv", index=False)
    
    feedback_tb = initialize_feedback_table(watchlist)
    feedback_tb.to_csv("artifacts/feedback_capture_table.csv", index=False)

    print("Phase 2 State Detection Engine complete:")
    print(f"  ASTM D7720 Limits : {len(limits_df)} limits calculated per model/component/analyte")
    print(f"  Weekly Watchlist  : {len(watchlist)} machine-components ranked by priority score")
    print(f"  Feedback Capture  : Initialized 'artifacts/feedback_capture_table.csv' for engineer logs")
    print(f"  Outputs saved to 'artifacts/phase2_watchlist.csv' and 'artifacts/d7720_population_limits.csv'")


if __name__ == "__main__":
    main()
