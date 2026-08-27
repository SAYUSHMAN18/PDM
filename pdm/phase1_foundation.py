"""
Phase 1 - Data Foundation & Quality Audit  (weeks 3-7)

One canonical dataset, and a numbers-based verdict on whether the labels can be
trusted.

  * Data-quality profile   null shares, meter monotonicity, join rates.
  * Label-governance audit  PM/CM split, WO -> compartment mapping share,
                            reporting-lag proxy. STOP-AND-FIX if mapping < 50%.
  * Sampling-discipline     share of samples carrying oil hours / fluid-changed,
                            sampling-interval distribution per family.
  * Physics normalisation   ppm -> wear-metal mass rate (mg / operating hour).

Outputs
  data/processed/canonical_samples.csv
  artifacts/phase1_data_audit.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .data import Extracts, load_extracts
from .features import add_labels, fit_peer_baseline, add_features


def _null_share(df: pd.DataFrame, cols) -> dict:
    return {c: round(float(df[c].isna().mean()), 3) for c in cols if c in df.columns}


def profile_quality(ext: Extracts) -> dict:
    s, w = ext.sos, ext.wo
    smu_seq = s.dropna(subset=["smu_hours"]).sort_values(["machine_id", "sample_date"])
    non_monotone = 0
    for _, g in smu_seq.groupby("machine_id"):
        non_monotone += int((g["smu_hours"].diff() < -1).sum())

    wo_machines = set(w["machine_id"]) if len(w) else set()
    return {
        "sos_rows": len(s),
        "wo_rows": len(w),
        "sos_null_share": _null_share(s, ["smu_hours", "oil_hours", "sample_date",
                                          "model_family"] + config.WEAR_METALS),
        "smu_meter_reversals": non_monotone,
        "sos_machines_with_a_work_order_pct": round(
            100 * len(set(s["machine_id"]) & wo_machines) / max(s["machine_id"].nunique(), 1), 1),
        "date_range": [str(s["sample_date"].min().date()), str(s["sample_date"].max().date())],
    }


def audit_label_governance(ext: Extracts) -> dict:
    w = ext.wo
    if len(w) == 0:
        return {"status": "NO_WORK_ORDERS", "stop_and_fix": True}

    total = len(w)
    cm = int((w["wo_type"] == "CM").sum())
    pm = int((w["wo_type"] == "PM").sum())
    mapped = int((w["component"] != "OTHER").sum())
    mapping_share = mapped / total

    # reporting-lag proxy: days from the last ACTION/CRITICAL sample to the CM WO
    lags = []
    sev = ext.sos[ext.sos["lab_severity"].isin(["ACTION", "CRITICAL"])]
    cm_wo = w[w["wo_type"] == "CM"]
    for _, r in cm_wo.iterrows():
        cand = sev[(sev["machine_id"] == r["machine_id"])
                   & (sev["component"] == r["component"])
                   & (sev["sample_date"] <= r["open_date"])]
        if len(cand):
            lags.append((r["open_date"] - cand["sample_date"].max()).days)

    return {
        "total_work_orders": total,
        "corrective": cm,
        "preventive": pm,
        "corrective_share": round(cm / total, 3),
        "wo_to_compartment_mapping_share": round(mapping_share, 3),
        "reporting_lag_days_median": float(np.median(lags)) if lags else None,
        "reporting_lag_days_p90": float(np.quantile(lags, 0.9)) if lags else None,
        "stop_and_fix": bool(mapping_share < 0.50 or cm == 0),
        "stop_and_fix_reason": (
            "WO->compartment mapping below 50%" if mapping_share < 0.50
            else "no corrective work orders" if cm == 0 else None),
    }


def audit_sampling_discipline(ext: Extracts) -> dict:
    s = ext.sos
    oil_ok = (s["oil_hours"].notna() & (s["oil_hours"] > 0)).mean()
    intervals = (s.sort_values("sample_date")
                 .groupby(["machine_id", "component"])["sample_date"].diff().dt.days.dropna())
    by_family = (s.assign(gap=s.sort_values("sample_date")
                          .groupby(["machine_id", "component"])["sample_date"].diff().dt.days)
                 .groupby("component_family")["gap"].median().round(1).to_dict())
    return {
        "samples_with_oil_hours_share": round(float(oil_ok), 3),
        "fluid_changed_flag_populated": bool(s["fluid_changed"].any()),
        "sampling_interval_days_median": float(intervals.median()) if len(intervals) else None,
        "sampling_interval_days_p90": float(intervals.quantile(0.9)) if len(intervals) else None,
        "median_interval_by_family": by_family,
        "rating": "GOOD" if oil_ok >= 0.70 else "NEEDS_IMPROVEMENT",
    }


def main(ext: Extracts | None = None) -> dict:
    config.ensure_dirs()
    ext = ext or load_extracts(verbose=False)

    quality = profile_quality(ext)
    labels = audit_label_governance(ext)
    sampling = audit_sampling_discipline(ext)

    # canonical modelling table (labels + physics-normalised features)
    labelled = add_labels(ext.sos, ext.wo, config.PRIMARY_HORIZON)
    cutoff = labelled["sample_date"].quantile(config.PEER_TRAIN_FRACTION)
    peer = fit_peer_baseline(labelled, cutoff)
    canonical = add_features(labelled, peer)
    canonical.to_csv(config.DATA_PROCESSED / "canonical_samples.csv", index=False)
    peer.to_csv(config.ARTIFACTS / "peer_baseline.csv", index=False)

    audit = {
        "phase": "1 - Data Foundation & Quality Audit",
        "data_mode": ext.mode,
        "data_quality": quality,
        "label_governance": labels,
        "sampling_discipline": sampling,
        "canonical_rows": len(canonical),
        "physics_mass_rate_columns": [f"{m}__mass_rate_mg_h" for m in config.WEAR_METALS],
    }
    with open(config.ARTIFACTS / "phase1_data_audit.json", "w") as f:
        json.dump(audit, f, indent=2, default=str)

    from . import report
    report.phase1_summary(
        n_samples=len(canonical),
        reversals=quality['smu_meter_reversals'],
        wo_map_pct=labels.get('wo_to_compartment_mapping_share'),
        cm_share=labels.get('corrective_share'),
        lag_median=labels.get('reporting_lag_days_median'),
        oil_hrs_pct=sampling['samples_with_oil_hours_share'],
        oil_rating=sampling['rating'],
        no_wo=labels.get('status') == 'NO_WORK_ORDERS',
        stop_fix=bool(labels.get('stop_and_fix', False)),
        stop_reason=labels.get('stop_and_fix_reason'),
    )
    return audit


if __name__ == "__main__":
    main()
