"""
Phase 5 — Advisory Generation & Deployment (Cost-Weighted Ranking)

Tasks:
  1. Cost-Weighted Ranking: Rank recommendations by Expected Cost Saved = Risk * (Downtime + Parts + Labour).
  2. Inspection Advisory generation for maintenance planners and CMMS integration.
  3. Value Case & ROI Calculation (Prevented failures * Avoided cost - Inspection cost).

Run:  python phase5_advisory_deployment.py
Out:  data/processed/phase5_cost_ranked_advisories.csv
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

from phase4_survival_prognostics import compute_multi_horizon_survival_risk
from build_dataset import load_raw


COMPONENT_COST_ESTIMATES = {
    "ENGINE": {"parts_usd": 35000, "labour_usd": 8000, "downtime_h": 48, "downtime_cost_per_h": 250},
    "TRANSMISSION": {"parts_usd": 22000, "labour_usd": 5000, "downtime_h": 36, "downtime_cost_per_h": 250},
    "HYDRAULIC": {"parts_usd": 15000, "labour_usd": 4000, "downtime_h": 24, "downtime_cost_per_h": 250},
    "DIFFERENTIAL": {"parts_usd": 18000, "labour_usd": 4500, "downtime_h": 30, "downtime_cost_per_h": 250},
    "FINAL_DRIVE": {"parts_usd": 12000, "labour_usd": 3000, "downtime_h": 20, "downtime_cost_per_h": 250},
    "WHEEL_END": {"parts_usd": 8000, "labour_usd": 2000, "downtime_h": 16, "downtime_cost_per_h": 250}
}


def calculate_cost_weighted_advisories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    for idx, r in df.iterrows():
        comp = r["component"]
        cost_info = COMPONENT_COST_ESTIMATES.get(comp, COMPONENT_COST_ESTIMATES["FINAL_DRIVE"])
        
        parts = cost_info["parts_usd"]
        labour = cost_info["labour_usd"]
        downtime = cost_info["downtime_h"] * cost_info["downtime_cost_per_h"]
        total_failure_cost = parts + labour + downtime
        
        risk_30d = r["risk_30d"]
        expected_cost_at_risk = risk_30d * total_failure_cost
        
        df.loc[idx, "total_failure_cost_usd"] = total_failure_cost
        df.loc[idx, "expected_cost_at_risk_usd"] = round(expected_cost_at_risk, 2)
        
        # Priority Rank: Cost-Weighted expected risk
        if expected_cost_at_risk >= 15000:
            df.loc[idx, "action_priority"] = "P1_IMMEDIATE_INSPECTION"
            df.loc[idx, "recommended_action"] = "Schedule immediate shutdown inspection & oil resample"
        elif expected_cost_at_risk >= 5000:
            df.loc[idx, "action_priority"] = "P2_PLANNED_MAINTENANCE"
            df.loc[idx, "recommended_action"] = "Inspect at next scheduled service interval"
        else:
            df.loc[idx, "action_priority"] = "P3_MONITOR"
            df.loc[idx, "recommended_action"] = "Continue routine sampling"
            
    return df.sort_values("expected_cost_at_risk_usd", ascending=False).reset_index(drop=True)


def calculate_value_case(advisories: pd.DataFrame, inspection_cost_usd=300) -> dict:
    high_risk_count = (advisories["action_priority"] == "P1_IMMEDIATE_INSPECTION").sum()
    total_prevented_cost = advisories[advisories["action_priority"] == "P1_IMMEDIATE_INSPECTION"]["expected_cost_at_risk_usd"].sum()
    total_inspection_cost = len(advisories) * inspection_cost_usd
    net_value_saved = total_prevented_cost - total_inspection_cost
    roi = (net_value_saved / total_inspection_cost * 100) if total_inspection_cost > 0 else 0.0

    return {
        "total_components_monitored": len(advisories),
        "high_risk_p1_alerts": int(high_risk_count),
        "total_prevented_cost_usd": float(total_prevented_cost),
        "total_inspection_cost_usd": float(total_inspection_cost),
        "net_value_saved_usd": float(net_value_saved),
        "estimated_roi_percent": float(roi)
    }


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    sos, wo = load_raw()
    survival_df = compute_multi_horizon_survival_risk(sos, wo)
    advisories = calculate_cost_weighted_advisories(survival_df)
    
    out_path = "data/processed/phase5_cost_ranked_advisories.csv"
    advisories.to_csv(out_path, index=False)
    
    val_case = calculate_value_case(advisories)

    print("Phase 5 Advisory Generation & Deployment complete:")
    print(f"  Cost-Weighted Advisories : {len(advisories)} machine-components ranked by Expected Cost Saved")
    print(f"  P1 Immediate Inspections  : {val_case['high_risk_p1_alerts']} critical components")
    print(f"  Estimated Net Value Saved : ${val_case['net_value_saved_usd']:,.2f}")
    print(f"  Estimated Pilot ROI       : {val_case['estimated_roi_percent']:.1f}%")
    print(f"  Output saved to           : '{out_path}'")
    print("\nTop Cost-Weighted Maintenance Advisories:")
    print(advisories[["machine_id", "component", "risk_30d", "expected_cost_at_risk_usd", "action_priority"]].head(5))


if __name__ == "__main__":
    main()
