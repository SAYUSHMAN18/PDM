"""
Phase 5 - Advisory Generation & Deployment  (weeks 20-27)

Turn model risk into a worklist a maintenance planner will actually action, and
start the feedback loop that lets the system learn from its own alerts.

  * Cost weighting     rank by  risk x detection-rate x failure-cost, not by
                       probability. A 40% risk on a final drive outranks a 70%
                       risk on something cheap -- and this is the version that
                       gets budget approved.
  * Failure cost       median (parts + labour + downtime) of real corrective work
                       orders for that component family, with a config fallback
                       when the history is too thin.
  * Advisory tiers     P1 inspect now / P2 plan into next service / P3 keep
                       sampling, from expected-value thresholds in config.
  * Value case         prevented cost vs inspection cost -> pilot ROI.
  * Feedback capture    every advisory gets an outcome row: confirmed /
                       not_confirmed / not_inspected. Appended, never overwritten.

Inputs (first that exists)
  artifacts/phase4_multi_horizon_risk.csv     (survival risk_30d)
  data/processed/risk_scores.csv              (Phase 3 classifier risk)

Outputs
  data/processed/phase5_cost_ranked_advisories.csv
  artifacts/phase5_value_case.json
  artifacts/advisory_feedback_table.csv
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from . import config
from .data import Extracts, load_extracts

P4_RISK = config.ARTIFACTS / "phase4_multi_horizon_risk.csv"
P3_RISK = config.DATA_PROCESSED / "risk_scores.csv"


# --------------------------------------------------------------------- cost model
def failure_cost_by_family(wo: pd.DataFrame) -> dict:
    """Median all-in cost of a corrective job per component family, config fallback."""
    out = {}
    cm = wo[wo["wo_type"] == "CM"] if len(wo) else wo
    for fam, spec in config.FAILURE_COST.items():
        fallback = spec["parts_usd"] + spec["labour_usd"] + spec["downtime_h"] * spec["downtime_usd_per_h"]
        rows = cm[cm["component_family"] == fam] if len(cm) else cm
        if len(rows) >= 3:
            allin = (rows["parts_cost"].fillna(spec["parts_usd"])
                     + rows["labour_cost"].fillna(spec["labour_usd"])
                     + rows["downtime_h"].fillna(spec["downtime_h"]) * spec["downtime_usd_per_h"])
            out[fam] = {"total_failure_cost_usd": round(float(allin.median()), 0),
                        "source": f"median of {len(rows)} corrective WOs"}
        else:
            out[fam] = {"total_failure_cost_usd": round(float(fallback), 0),
                        "source": "config fallback (history too thin)"}
    return out


# ------------------------------------------------------------------- risk loading
def load_risk() -> tuple[pd.DataFrame, str]:
    if P4_RISK.exists():
        df = pd.read_csv(P4_RISK, parse_dates=["sample_date"])
        df = df.rename(columns={"risk_30d": "risk"})
        return df, "phase4_survival"
    if P3_RISK.exists():
        df = pd.read_csv(P3_RISK, parse_dates=["sample_date"])
        return df, "phase3_classifier"
    raise FileNotFoundError("Run Phase 3 and/or Phase 4 first -- no risk scores to rank.")


# --------------------------------------------------------------------- advisories
_TIER_ACTION = {
    "P1": "Inspect now; take a confirmation oil sample before further operation.",
    "P2": "Book an inspection into the next scheduled service for this machine.",
    "P3": "No action; continue routine sampling and watch the trend.",
}


def build_advisories(risk: pd.DataFrame, costs: dict) -> pd.DataFrame:
    df = risk.copy()
    if "component_family" not in df.columns:
        df["component_family"] = df["component"].map(config.compartment_family)

    df["total_failure_cost_usd"] = df["component_family"].map(
        lambda f: costs.get(f, costs.get("OTHER"))["total_failure_cost_usd"])
    df["expected_value_usd"] = (df["risk"] * config.DETECTION_RATE
                                * df["total_failure_cost_usd"]).round(0)
    df["net_value_usd"] = (df["expected_value_usd"] - config.INSPECTION_COST_USD).round(0)

    df["tier"] = np.where(df["expected_value_usd"] >= config.P1_EXPECTED_VALUE_USD, "P1",
                 np.where(df["expected_value_usd"] >= config.P2_EXPECTED_VALUE_USD, "P2", "P3"))
    df["recommended_action"] = df["tier"].map(_TIER_ACTION)

    cols = ["machine_id", "model", "model_family", "component", "component_family",
            "position", "sample_date", "lab_severity", "risk",
            "total_failure_cost_usd", "expected_value_usd", "net_value_usd",
            "tier", "recommended_action"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].sort_values("expected_value_usd", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def value_case(adv: pd.DataFrame) -> dict:
    actioned = adv[adv["tier"].isin(["P1", "P2"])]
    prevented = float((actioned["expected_value_usd"]).sum())
    inspection_cost = len(actioned) * config.INSPECTION_COST_USD
    net = prevented - inspection_cost
    return {
        "components_monitored": int(len(adv)),
        "p1_inspect_now": int((adv["tier"] == "P1").sum()),
        "p2_plan_inspection": int((adv["tier"] == "P2").sum()),
        "p3_monitor": int((adv["tier"] == "P3").sum()),
        "expected_prevented_cost_usd": round(prevented, 0),
        "inspection_cost_usd": round(float(inspection_cost), 0),
        "net_expected_value_usd": round(net, 0),
        "roi_x": round(net / inspection_cost, 2) if inspection_cost else None,
        "assumptions": {
            "detection_rate": config.DETECTION_RATE,
            "inspection_cost_usd": config.INSPECTION_COST_USD,
            "p1_threshold_usd": config.P1_EXPECTED_VALUE_USD,
            "p2_threshold_usd": config.P2_EXPECTED_VALUE_USD,
        },
    }


def update_feedback_table(adv: pd.DataFrame) -> pd.DataFrame:
    path = config.ARTIFACTS / "advisory_feedback_table.csv"
    new = pd.DataFrame({
        "advisory_id": (adv["machine_id"].astype(str) + "|" + adv["component"].astype(str)
                        + "|" + pd.to_datetime(adv["sample_date"]).dt.strftime("%Y%m%d")),
        "machine_id": adv["machine_id"], "component": adv["component"],
        "sample_date": pd.to_datetime(adv["sample_date"]).dt.date,
        "tier": adv["tier"], "risk": adv["risk"].round(3),
        "expected_value_usd": adv["expected_value_usd"],
        "advisory_date": pd.Timestamp.today().date(),
        "inspection_outcome": "not_inspected",   # -> confirmed / not_confirmed / not_inspected
        "actual_finding": "", "reviewed_by": "", "review_date": "",
    })
    if path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old, new[~new["advisory_id"].isin(old["advisory_id"])]],
                           ignore_index=True)
    else:
        merged = new
    merged.to_csv(path, index=False)
    return merged


# --------------------------------------------------------------------------- main
def main(ext: Extracts | None = None) -> dict:
    config.ensure_dirs()
    ext = ext or load_extracts(verbose=False)

    risk, source = load_risk()
    costs = failure_cost_by_family(ext.wo)
    adv = build_advisories(risk, costs)
    adv.to_csv(config.DATA_PROCESSED / "phase5_cost_ranked_advisories.csv", index=False)

    vc = value_case(adv)
    vc["risk_source"] = source
    with open(config.ARTIFACTS / "phase5_value_case.json", "w") as f:
        json.dump(vc, f, indent=2, default=str)
    feedback = update_feedback_table(adv)

    print("Phase 5 - Advisory Generation & Cost-Weighted Deployment")
    print(f"  risk source           : {source}")
    print(f"  advisories            : {len(adv)}  "
          f"(P1 {vc['p1_inspect_now']} / P2 {vc['p2_plan_inspection']} / P3 {vc['p3_monitor']})")
    print(f"  expected prevented $  : ${vc['expected_prevented_cost_usd']:,.0f}")
    print(f"  inspection cost $     : ${vc['inspection_cost_usd']:,.0f}")
    print(f"  net expected value $  : ${vc['net_expected_value_usd']:,.0f}  (ROI {vc['roi_x']}x)")
    print(f"  feedback table        : {len(feedback)} advisories tracked  ->  artifacts/advisory_feedback_table.csv")
    print("\n  Top cost-weighted advisories:")
    for _, r in adv.head(8).iterrows():
        print(f"   {r['rank']:>2}. {r['machine_id']:<12} {r['component']:<10} "
              f"{r['tier']}  risk {r['risk']:>5.0%}  E[value] ${r['expected_value_usd']:>10,.0f}  "
              f"{r['component_family']}")
    return vc


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
