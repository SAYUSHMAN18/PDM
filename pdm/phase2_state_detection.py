"""
Phase 2 - State Detection: the first thing that ships  (weeks 6-11)

A weekly ranked watchlist in front of real engineers, built with NO work orders.

  * ASTM D7720 style limits   fleet-population percentiles per (model family,
                              component family, analyte), compared against the
                              generic OEM condemning limits.
  * Time-aware change signal  EWMA deviation and local slope from features.py,
                              charted against oil hours and reset at each oil
                              change (never alarms on a fresh-oil dilution).
  * Multivariate novelty      Mahalanobis distance of the latest sample from the
                              healthy wear-metal covariance of its peer group.
  * InterpText mining         the lab's prose diagnosis -> structured flags.
  * Weekly watchlist          composite priority score, ranked.
  * Feedback capture table    every alert gets an outcome: confirmed /
                              not_confirmed / not_inspected. Appended, never
                              overwritten -- six months of this is your label set.

Outputs
  data/processed/phase2_watchlist.csv
  artifacts/d7720_population_limits.csv
  artifacts/feedback_capture_table.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import Extracts, load_extracts
from .features import add_labels, fit_peer_baseline, add_features

ANALYTES = config.ANALYTES


# ------------------------------------------------------------- D7720 style limits
def compute_population_limits(sos: pd.DataFrame) -> pd.DataFrame:
    healthy = sos[sos["lab_severity"].isin(config.HEALTHY_SEVERITIES)]
    if len(healthy) < 20:
        healthy = sos
    rows = []
    for (mf, cf), g in healthy.groupby(["model_family", "component_family"]):
        generic = config.GENERIC_CONDEMNING_LIMITS.get(cf, {})
        for a in ANALYTES:
            v = g[a].dropna()
            if len(v) < 8:
                continue
            q90, q95, q99 = np.quantile(v, [0.90, 0.95, 0.99])
            gl = generic.get(a, np.nan)
            rows.append({
                "model_family": mf, "component_family": cf, "analyte": a,
                "n": len(v), "median": round(float(v.median()), 3),
                "watch_p90": round(float(q90), 3),
                "action_p95": round(float(q95), 3),
                "critical_p99": round(float(q99), 3),
                "generic_condemning_limit": gl,
                "fleet_limit_tighter_than_generic": bool(np.isfinite(gl) and q95 < gl),
            })
    return pd.DataFrame(rows)


def _limit_exceedances(latest: pd.DataFrame, limits: pd.DataFrame) -> pd.Series:
    if limits.empty:
        return pd.Series(0, index=latest.index)
    lut = limits.set_index(["model_family", "component_family", "analyte"])
    counts = np.zeros(len(latest))
    for i, (_, r) in enumerate(latest.iterrows()):
        for a in config.HIGHER_IS_WORSE:
            try:
                lim = lut.loc[(r["model_family"], r["component_family"], a), "action_p95"]
            except KeyError:
                continue
            if pd.notna(r[a]) and pd.notna(lim) and r[a] > lim:
                counts[i] += 1
    return pd.Series(counts, index=latest.index)


# ------------------------------------------------------------- novelty detection
def mahalanobis_novelty(sos: pd.DataFrame, latest: pd.DataFrame) -> pd.Series:
    metals = [m for m in config.WEAR_METALS if sos[m].notna().any()]
    if len(metals) < 2:
        return pd.Series(0.0, index=latest.index)
    out = pd.Series(0.0, index=latest.index)
    for cf, g in latest.groupby("component_family"):
        healthy = sos[(sos["component_family"] == cf)
                      & sos["lab_severity"].isin(config.HEALTHY_SEVERITIES)][metals].dropna()
        if len(healthy) < len(metals) + 5:
            healthy = sos[sos["component_family"] == cf][metals].dropna()
        if len(healthy) < len(metals) + 2:
            continue
        mu = healthy.mean().to_numpy()
        cov = np.cov(healthy.to_numpy(), rowvar=False)
        inv = np.linalg.pinv(cov + np.eye(len(metals)) * 1e-6)
        x = g[metals].fillna(healthy.mean()).to_numpy()
        d = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", x - mu, inv, x - mu), 0, None))
        out.loc[g.index] = d
    return out


# --------------------------------------------------------------------- watchlist
def build_watchlist(full: pd.DataFrame, sos: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
    latest = (full.sort_values("sample_date")
              .groupby(["machine_id", "component"], as_index=False).tail(1)
              .reset_index(drop=True))

    latest["novelty_score"] = mahalanobis_novelty(sos, latest)
    latest["d7720_action_exceedances"] = _limit_exceedances(latest, limits)

    ewma_dev_cols = [f"{m}__ewma_dev" for m in config.WEAR_METALS if f"{m}__ewma_dev" in latest.columns]
    latest["wear_accel"] = latest[ewma_dev_cols].clip(lower=0).sum(axis=1) if ewma_dev_cols else 0.0
    nov = latest["novelty_score"]
    latest["novelty_z"] = (nov - nov.median()) / (nov.std(ddof=0) or 1.0)

    latest["priority_score"] = (
        latest["lab_severity_num"] * 25
        + latest["novelty_z"].clip(lower=0) * 12
        + latest["d7720_action_exceedances"] * 10
        + np.log1p(latest["wear_accel"]) * 6
        + latest.get("flag__coolant", 0) * 20
        + latest.get("flag__water", 0) * 12
        + latest.get("flag__dirt", 0) * 10
        + latest.get("flag__bearing", 0) * 10
        + latest.get("flag__iron", 0) * 6
    ).round(2)

    cols = ["machine_id", "model", "model_family", "component", "component_family",
            "position", "sample_date", "smu_hours", "oil_hours", "lab_severity",
            "lab_severity_num", "novelty_score", "novelty_z", "d7720_action_exceedances",
            "wear_accel", "priority_score"]
    flag_cols = [c for c in latest.columns if c.startswith("flag__")]
    ranked = (latest[cols + flag_cols]
              .sort_values("priority_score", ascending=False)
              .reset_index(drop=True))
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def update_feedback_table(watchlist: pd.DataFrame) -> pd.DataFrame:
    path = config.ARTIFACTS / "feedback_capture_table.csv"
    new = pd.DataFrame({
        "alert_id": watchlist["machine_id"].astype(str) + "|" + watchlist["component"].astype(str)
        + "|" + watchlist["sample_date"].dt.strftime("%Y%m%d"),
        "machine_id": watchlist["machine_id"],
        "component": watchlist["component"],
        "sample_date": watchlist["sample_date"].dt.date,
        "priority_score": watchlist["priority_score"],
        "lab_severity": watchlist["lab_severity"],
        "alert_date": pd.Timestamp.today().date(),
        "inspection_outcome": "not_inspected",   # -> confirmed / not_confirmed / not_inspected
        "inspection_notes": "",
        "reviewed_by": "",
        "review_date": "",
    })
    if path.exists():
        old = pd.read_csv(path)
        merged = (pd.concat([old, new[~new["alert_id"].isin(old["alert_id"])]], ignore_index=True))
    else:
        merged = new
    merged.to_csv(path, index=False)
    return merged


def main(ext: Extracts | None = None) -> dict:
    config.ensure_dirs()
    ext = ext or load_extracts(verbose=False)

    labelled = add_labels(ext.sos, ext.wo, config.PRIMARY_HORIZON)
    cutoff = labelled["sample_date"].quantile(config.PEER_TRAIN_FRACTION)
    peer = fit_peer_baseline(labelled, cutoff)
    full = add_features(labelled, peer)

    limits = compute_population_limits(ext.sos)
    limits.to_csv(config.ARTIFACTS / "d7720_population_limits.csv", index=False)

    watchlist = build_watchlist(full, ext.sos, limits)
    watchlist.to_csv(config.DATA_PROCESSED / "phase2_watchlist.csv", index=False)
    feedback = update_feedback_table(watchlist)

    tighter = int(limits["fleet_limit_tighter_than_generic"].sum()) if len(limits) else 0
    top = watchlist.head(10)

    from . import report
    report.phase2_summary(
        n_limits=len(limits),
        n_tighter=tighter,
        n_watchlist=len(watchlist),
        no_chemistry=not ext.has_chemistry,
        top_rows=top.to_dict(orient="records"),
    )
    return {"limits": len(limits), "watchlist": len(watchlist), "feedback_alerts": len(feedback)}


if __name__ == "__main__":
    main()
