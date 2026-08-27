"""
Phase 1 — Data Foundation & Quality Audit

Tasks:
  1. Profile raw extracts and canonicalize schema.
  2. Label governance audit: PM/CM flag reliability, reporting lag, WO-to-compartment mapping share (~50% stop-and-fix condition).
  3. Sampling discipline audit: share of samples carrying oil hours and fluid-changed flags.
  4. Physics Normalization: Convert PPM concentrations to wear-metal mass rates per hour (mass_rate = (ppm * sump_volume_l) / oil_hours).

Run:  python phase1_foundation.py
Out:  data/processed/canonical_samples.csv
      artifacts/phase1_data_audit.json
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

from build_dataset import ANALYTES, load_raw
from phase0_scope import COMPARTMENT_TAXONOMY


def audit_label_governance(wo: pd.DataFrame) -> dict:
    total_wo = len(wo)
    if total_wo == 0:
        return {"cmms_reliability_score": 0.0, "status": "NO_WORK_ORDERS_FOUND"}
    
    cm_count = (wo["wo_type"] == "CM").sum()
    pm_count = (wo["wo_type"] == "PM").sum()
    mapped_compartments = (wo["component"] != "OTHER").sum()
    mapping_share = mapped_compartments / total_wo
    
    reporting_lag_days = []
    if "open_date" in wo.columns and "close_date" in wo.columns:
        wo["open_date"] = pd.to_datetime(wo["open_date"])
        wo["close_date"] = pd.to_datetime(wo["close_date"])
        lags = (wo["close_date"] - wo["open_date"]).dt.days
        reporting_lag_days = lags.dropna().tolist()

    avg_lag = float(np.mean(reporting_lag_days)) if reporting_lag_days else 0.0

    return {
        "total_work_orders": int(total_wo),
        "corrective_wo_count": int(cm_count),
        "preventive_wo_count": int(pm_count),
        "wo_to_compartment_mapping_share": float(mapping_share),
        "avg_reporting_lag_days": avg_lag,
        "stop_and_fix_condition_triggered": bool(mapping_share < 0.50)
    }


def audit_sampling_discipline(sos: pd.DataFrame) -> dict:
    total_samples = len(sos)
    if total_samples == 0:
        return {"status": "NO_SAMPLES_FOUND"}

    oil_hours_populated = sos["oil_hours"].notna() & (sos["oil_hours"] > 0)
    oil_hours_share = float(oil_hours_populated.mean())

    fluid_changed_populated = "oil_changed" in sos.columns or "FluidChanged" in sos.columns
    
    return {
        "total_samples": int(total_samples),
        "samples_with_valid_oil_hours_share": oil_hours_share,
        "fluid_changed_flag_tracked": fluid_changed_populated,
        "sampling_discipline_rating": "GOOD" if oil_hours_share >= 0.70 else "NEEDS_IMPROVEMENT"
    }


def normalize_physics_wear(sos: pd.DataFrame) -> pd.DataFrame:
    df = sos.copy()
    
    for idx, row in df.iterrows():
        comp = row["component"]
        tax = COMPARTMENT_TAXONOMY.get(comp, {"oil_volume_l": 50.0})
        vol_l = tax["oil_volume_l"]
        oil_h = max(row["oil_hours"], 1.0) if pd.notna(row["oil_hours"]) else 250.0
        smu_h = max(row["smu_hours"], 1.0) if pd.notna(row["smu_hours"]) else 1000.0

        # Calculate Mass Wear Rate: mg metal generated per operating hour
        # (concentration PPM = mg/L -> total mg = ppm * volume_l -> wear_rate_mg_per_h = (ppm * vol_l) / oil_hours)
        for metal in ["fe_ppm", "cu_ppm", "cr_ppm", "pb_ppm", "al_ppm", "si_ppm"]:
            if metal in df.columns:
                ppm = row[metal]
                mass_mg = ppm * vol_l
                df.loc[idx, f"{metal}__mass_rate_mg_h"] = mass_mg / oil_h
                df.loc[idx, f"{metal}__per100h"] = (ppm / oil_h) * 100.0

    return df


def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    sos, wo = load_raw()
    
    label_audit = audit_label_governance(wo)
    sample_audit = audit_sampling_discipline(sos)
    
    canonical_sos = normalize_physics_wear(sos)
    canonical_sos.to_csv("data/processed/canonical_samples.csv", index=False)

    audit_summary = {
        "phase": "Phase 1 - Data Foundation & Quality Audit",
        "label_governance_audit": label_audit,
        "sampling_discipline_audit": sample_audit,
        "canonical_sample_count": len(canonical_sos),
        "physics_mass_rates_calculated": True
    }

    with open("artifacts/phase1_data_audit.json", "w") as f:
        json.dump(audit_summary, f, indent=2)

    print("Phase 1 Data Foundation & Quality Audit complete:")
    print(f"  Canonical Samples : {len(canonical_sos)} rows saved to 'data/processed/canonical_samples.csv'")
    print(f"  WO Mapping Share  : {label_audit.get('wo_to_compartment_mapping_share', 0.0):.1%}")
    print(f"  Oil Hours Share   : {sample_audit.get('samples_with_valid_oil_hours_share', 0.0):.1%}")
    print(f"  Audit output saved: 'artifacts/phase1_data_audit.json'")


if __name__ == "__main__":
    main()
