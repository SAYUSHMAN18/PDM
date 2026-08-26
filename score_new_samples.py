"""
Score the latest oil sample for every machine + component and print risk cards
an engineer can act on.

Run:  python score_new_samples.py [--top 10]
Out:  data/processed/risk_scores.csv  (+ printed cards)
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import load

from build_dataset import ANALYTES, add_features, add_labels, load_raw

PRETTY = {
    "fe_ppm": "iron", "cu_ppm": "copper", "cr_ppm": "chromium", "pb_ppm": "lead",
    "al_ppm": "aluminium", "si_ppm": "silicon (dirt)", "na_ppm": "sodium", "k_ppm": "potassium",
    "water_pct": "water", "fuel_pct": "fuel dilution", "glycol_pct": "glycol (coolant)",
    "soot_pct": "soot", "visc40": "viscosity @40C", "oxidation": "oxidation",
    "nitration": "nitration", "tbn": "TBN (additive reserve)", "pq_index": "PQ (ferrous debris)",
}
SUFFIX = {
    "__peer_z": "above the fleet norm", "__delta": "up since the last sample",
    "__pct_change": "rising", "__vs_own_med": "above this machine's own baseline",
    "__slope3": "trending up over 3 samples", "__per100h": "wearing fast per 100 hours",
}


def describe(feature: str) -> str:
    for suf, phrase in SUFFIX.items():
        if feature.endswith(suf):
            return f"{PRETTY.get(feature[:-len(suf)], feature[:-len(suf)])} {phrase}"
    return PRETTY.get(feature, feature.replace("_", " "))


def contributions(bundle, x_row: pd.DataFrame, top=4) -> list[str]:
    """Why this sample scored high. Exact for the linear model, importance-weighted otherwise."""
    model, feats = bundle["model"], bundle["features"]
    try:                                   # linear pipeline -> exact per-feature push
        imp, sc, clf = model.named_steps["impute"], model.named_steps["scale"], model.named_steps["clf"]
        z = sc.transform(imp.transform(x_row[feats]))[0]
        push = z * clf.coef_[0]
    except (AttributeError, KeyError):     # tree model -> importance x how abnormal the value is
        fi = pd.read_csv("artifacts/feature_importance.csv").set_index("feature")["importance"]
        vals = x_row[feats].iloc[0]
        med = pd.read_csv("data/processed/model_table.csv", usecols=feats).median()
        spread = (med.abs() + 1e-6)
        push = (fi.reindex(feats).fillna(0).values
                * np.nan_to_num(((vals - med) / spread).values))
    out, seen = [], set()
    for i in np.argsort(-push):                       # one line per analyte, strongest first
        if push[i] <= 0:
            break
        base = feats[i].split("__")[0]
        if base in seen:
            continue
        seen.add(base)
        out.append(describe(feats[i]))
        if len(out) == top:
            break
    return out


def main(top: int, horizon_note: int = 30):
    sos, wo = load_raw()
    peer = pd.read_csv("artifacts/peer_stats.csv")
    bundle = load("artifacts/model.joblib")

    df = add_features(add_labels(sos, wo), peer)          # full history -> trend features
    latest = (df.sort_values("sample_date")
                .groupby(["machine_id", "component"], as_index=False).tail(1)
                .reset_index(drop=True))

    X = latest[bundle["features"]]
    latest["risk"] = bundle["model"].predict_proba(X)[:, 1]
    latest["risk_level"] = pd.cut(latest["risk"], [-1, bundle["threshold"] * 0.5,
                                                   bundle["threshold"], 0.5, 1.1],
                                  labels=["LOW", "WATCH", "HIGH", "CRITICAL"])

    out_cols = ["machine_id", "machine_type", "component", "sample_date", "smu_hours",
                "oil_hours", "lab_severity", "risk", "risk_level"]
    ranked = latest.sort_values("risk", ascending=False)
    ranked[out_cols].to_csv("data/processed/risk_scores.csv", index=False)

    print(f"\nScored {len(latest)} machine-components. "
          f"Alert threshold {bundle['threshold']:.3f} ({bundle['name']}).\n")
    for _, r in ranked.head(top).iterrows():
        drivers = contributions(bundle, latest.loc[[r.name]])
        print("=" * 62)
        print(f"Machine   : {r.machine_id}  ({r.machine_type})")
        print(f"Component : {r.component}")
        print(f"Sample    : {r.sample_date.date()}   SMU {r.smu_hours:,.0f} h   "
              f"oil age {r.oil_hours:,.0f} h   lab flag: {r.lab_severity}")
        print(f"Risk of corrective work order within {horizon_note} days: "
              f"{r.risk:6.1%}   [{r.risk_level}]")
        print("Main indicators:")
        for d in drivers or ["no single dominant indicator - driven by combined drift"]:
            print(f"  - {d}")
        print("Suggested action: "
              + ("inspect now and take a confirmation sample" if r.risk_level in ("HIGH", "CRITICAL")
                 else "re-sample at the next scheduled interval and watch the trend"))
    print("=" * 62)
    print("\nFull ranking written to data/processed/risk_scores.csv")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    main(ap.parse_args().top)
