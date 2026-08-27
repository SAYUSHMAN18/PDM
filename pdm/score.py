"""
Score the latest oil sample for every machine + compartment with the trained
Phase 3 model and print risk cards an engineer can act on.

    python -m pdm.score                 # score the latest sample in data/raw/
    python -m pdm.score --top 15
    python -m pdm.score --input path/to/new_sos_samples.csv

A --input file must use the same column contract as data/raw/sos_samples.csv
(see DATA_REQUEST.md). Trend features need a machine's history, so its earlier
samples should be in the file too, not just the new row.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import load

from . import config
from .data import load_extracts
from .features import add_features, add_labels, fit_peer_baseline
from .phase3_supervised import MODEL_PATH, _proba

_SUFFIX_PHRASE = {
    "__peer_z": "above the fleet norm", "__ewma_dev": "above its own smoothed trend",
    "__delta": "up since the last sample", "__pct_change": "rising",
    "__vs_own_med": "above this machine's own baseline",
    "__slope3": "trending up over three samples",
    "__per100h": "wearing fast per 100 hours",
    "__mass_rate_mg_h": "generating wear metal fast per operating hour",
}


_PLAIN = {"fe_per_oil_hour": "iron rising per oil hour",
          "si_to_fe": "silicon-to-iron ratio elevated (dirt vs wear)",
          "cu_to_fe": "copper-to-iron ratio elevated"}


def _describe(feat: str) -> str:
    for suf, phrase in _SUFFIX_PHRASE.items():
        if feat.endswith(suf):
            return f"{config.ANALYTE_PRETTY.get(feat[:-len(suf)], feat[:-len(suf)])} {phrase}"
    if feat in _PLAIN:
        return _PLAIN[feat]
    if feat in config.ANALYTE_PRETTY:
        return f"{config.ANALYTE_PRETTY[feat]} elevated"
    return feat.replace("_", " ")


# bookkeeping columns that are real model inputs but make poor "why" lines
_NOT_AN_INDICATOR = {
    "smu_hours", "oil_hours", "makeup_fluid_l", "sample_seq", "run_seq",
    "days_since_prev_sample", "hours_since_prev_sample", "days_since_last_cm",
    "oil_age_frac",
}


def _drivers(bundle: dict, row: pd.Series, pop: pd.DataFrame | None, top: int = 4) -> list[str]:
    """Top contributing signals. Exact for the linear model; importance x how
    abnormal the value is (vs the fleet) for the tree model."""
    feats = bundle["features"]
    est = bundle["model"]
    try:
        base = est.calibrated_classifiers_[0].estimator if hasattr(est, "calibrated_classifiers_") else est
        imp, sc, clf = base.named_steps["impute"], base.named_steps["scale"], base.named_steps["clf"]
        z = sc.transform(imp.transform(row[feats].to_frame().T))[0]
        push = z * clf.coef_[0]
    except Exception:
        try:
            fi = pd.read_csv(config.ARTIFACTS / "feature_importance.csv").set_index("feature")["importance"]
        except Exception:
            fi = pd.Series(1.0, index=feats)
        vals = row[feats].astype(float)
        if pop is not None:
            med = pop.reindex(columns=feats).median(numeric_only=True)
            iqr = (pop.reindex(columns=feats).quantile(0.75, numeric_only=True)
                   - pop.reindex(columns=feats).quantile(0.25, numeric_only=True)).replace(0, np.nan)
            abn = ((vals - med) / iqr).reindex(feats).to_numpy()
        else:
            abn = vals.to_numpy()
        weight = fi.reindex(feats).fillna(0).to_numpy()
        keep = np.array([f not in _NOT_AN_INDICATOR for f in feats])
        push = weight * np.nan_to_num(abn) * keep
    out, seen = [], set()
    for i in np.argsort(-push):
        if push[i] <= 0:
            break
        stem = feats[i].split("__")[0]
        if stem in seen:
            continue
        seen.add(stem)
        out.append(_describe(feats[i]))
        if len(out) >= top:
            break
    return out


def main(top: int = 10, input_csv: str | None = None) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model. Run:  python run_pipeline.py --phase 3")
    bundle = load(MODEL_PATH)

    if input_csv:
        from .data import _canonicalise_sos, _canonicalise_wo, _load_assets
        sos = _canonicalise_sos(pd.read_csv(input_csv), _load_assets())
        wo = _canonicalise_wo(None)
    else:
        ext = load_extracts(verbose=False)
        sos, wo = ext.sos, ext.wo

    labelled = add_labels(sos, wo, bundle["horizon_days"])
    peer = bundle.get("peer_baseline")
    if peer is None or len(peer) == 0:
        peer = fit_peer_baseline(labelled, labelled["sample_date"].quantile(config.PEER_TRAIN_FRACTION))
    full = add_features(labelled, peer)

    latest = (full.sort_values("sample_date")
              .groupby(["machine_id", "component"], as_index=False).tail(1)
              .reset_index(drop=True))
    latest["risk"] = _proba(bundle["model"], latest[bundle["features"]])
    thr = bundle["threshold"]
    latest["risk_level"] = pd.cut(latest["risk"], [-1, max(0.01, thr * 0.5), thr, 0.75, 1.1],
                                  labels=["LOW", "WATCH", "HIGH", "CRITICAL"], duplicates="drop")
    ranked = latest.sort_values("risk", ascending=False).reset_index(drop=True)

    out_cols = ["machine_id", "model", "component", "component_family", "sample_date",
                "smu_hours", "oil_hours", "lab_severity", "risk", "risk_level"]
    ranked[out_cols].to_csv(config.DATA_PROCESSED / "risk_scores.csv", index=False)

    h = bundle["horizon_days"]
    print(f"\nScored {len(ranked)} machine-compartments  "
          f"(model: {bundle['name']}, alert threshold {thr:.3f})\n")
    for _, r in ranked.head(top).iterrows():
        print("=" * 64)
        print(f"Machine   : {r['machine_id']}   ({r.get('model', '?')})")
        print(f"Component : {r['component']}  [{r['component_family']}]")
        smu = r["smu_hours"] if pd.notna(r["smu_hours"]) else float("nan")
        oh = r["oil_hours"] if pd.notna(r["oil_hours"]) else float("nan")
        print(f"Sample    : {pd.to_datetime(r['sample_date']).date()}   "
              f"SMU {smu:,.0f} h   oil age {oh:,.0f} h   lab flag: {r['lab_severity']}")
        print(f"Risk of corrective work order within {h} days: {r['risk']:.1%}   [{r['risk_level']}]")
        print("Main indicators:")
        for d in _drivers(bundle, r, full) or ["combined drift, no single dominant signal"]:
            print(f"  - {d}")
        act = ("inspect now and take a confirmation sample"
               if r["risk_level"] in ("HIGH", "CRITICAL")
               else "re-sample at the next interval and watch the trend")
        print(f"Suggested action: {act}")
    print("=" * 64)
    print("\nFull ranking -> data/processed/risk_scores.csv")
    return ranked


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--input", default=None)
    a = ap.parse_args()
    main(a.top, a.input)
