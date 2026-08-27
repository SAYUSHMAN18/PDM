"""
Phase 4 - Prognostics: survival with frailty  (weeks 16-22)

Risk of a corrective work order at 30 / 60 / 90 days from a single fit, using the
censored histories properly instead of a fixed classification window.

Method -- discrete-time survival via pooled logistic regression
  Each oil sample is expanded into person-period rows, one per 15-day interval out
  to the 90-day administrative censoring horizon. A row is "at risk" while the
  compartment has neither failed nor been censored; its event flag is 1 in the
  interval that contains the corrective work order. A single logistic regression
  on those rows estimates the per-interval hazard h(t | x); survival is the
  running product of (1 - h), and risk at a horizon is 1 - S(horizon).

  This is the standard person-period trick and it buys three things a plain
  classifier does not: right-censored samples still contribute the time they were
  observed, the censoring horizon is an explicit knob, and every horizon comes
  from one model.

Frailty
  Machines are repaired repeatedly, which breaks the independence a plain model
  assumes. We add a per-machine frailty proxy: the machine's own corrective-WO
  rate over the trailing year (computed from work orders strictly before the
  sample), shrunk toward the fleet mean. It is a random-effect stand-in -- real
  shared-frailty / Andersen-Gill fitting is a Phase 6 upgrade.

Outputs
  artifacts/phase4_multi_horizon_risk.csv    latest sample per machine-compartment
  artifacts/phase4_metrics.txt               fit summary + held-out check
  artifacts/phase4_km_baseline.csv           Kaplan-Meier baseline survival curve
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from . import config, model
from .data import Extracts, load_extracts
from .features import add_features, add_labels, fit_peer_baseline

CENSOR = config.SURVIVAL_CENSOR_D
STEP = config.SURVIVAL_STEP_D
N_PERIODS = CENSOR // STEP

# compact, physically-motivated feature set for the hazard model
_HAZARD_FEATURES = (
    [f"{m}__peer_z" for m in config.WEAR_METALS]
    + [f"{m}__ewma_dev" for m in config.WEAR_METALS]
    + [f"{m}__mass_rate_mg_h" for m in config.WEAR_METALS]
    + ["fe_ppm", "si_ppm", "pq_index", "water_pct", "glycol_pct",
       "lab_severity_num", "prior_cm_365d", "oil_age_frac", "si_to_fe",
       "machine_frailty"]
)


# ------------------------------------------------------------------- frailty proxy
def machine_frailty(sos: pd.DataFrame, wo: pd.DataFrame) -> pd.Series:
    """Per-sample trailing-year corrective-WO count for the machine, shrunk toward
    the fleet mean (empirical-Bayes style partial pooling)."""
    cm = wo[wo["wo_type"] == "CM"]
    if len(cm) == 0:
        return pd.Series(0.0, index=sos.index)
    by_machine = {m: np.sort(g["open_date"].values) for m, g in cm.groupby("machine_id")}
    fleet_rate = len(cm) / max(sos["machine_id"].nunique(), 1)
    prior_strength = 5.0
    out = np.zeros(len(sos))
    for i, (m, d) in enumerate(zip(sos["machine_id"].values, sos["sample_date"].values)):
        arr = by_machine.get(m)
        if arr is None:
            out[i] = fleet_rate / (1 + prior_strength)
            continue
        lo = np.searchsorted(arr, d - np.timedelta64(365, "D"), side="left")
        hi = np.searchsorted(arr, d, side="right")
        count = hi - lo
        out[i] = (count + fleet_rate) / (1 + prior_strength)
    return pd.Series(out, index=sos.index)


# -------------------------------------------------------------- person-period data
def to_person_period(df: pd.DataFrame, data_cutoff: pd.Timestamp) -> pd.DataFrame:
    """Expand each sample into <=N_PERIODS at-risk interval rows."""
    days_to_cutoff = (data_cutoff - df["sample_date"]).dt.days.clip(lower=0)
    censor_time = np.minimum(days_to_cutoff, CENSOR).to_numpy()
    dnext = df["days_to_next_cm"].to_numpy()
    observed = np.isfinite(dnext) & (dnext > 0) & (dnext <= censor_time)
    event_time = np.where(observed, dnext, censor_time)

    base_cols = [c for c in _HAZARD_FEATURES if c in df.columns]
    rows = []
    idx = df.index.to_numpy()
    for r in range(len(df)):
        et, obs = event_time[r], observed[r]
        for p in range(N_PERIODS):
            t0, t1 = p * STEP, (p + 1) * STEP
            if et <= t0:                       # already left the risk set
                break
            ev = int(obs and t0 < et <= t1)
            rows.append((idx[r], p, ev))
    pp = pd.DataFrame(rows, columns=["_src", "period", "event"])
    pp = pp.join(df[base_cols], on="_src")
    for p in range(N_PERIODS):
        pp[f"period_{p}"] = (pp["period"] == p).astype(int)
    return pp, observed, event_time


def _feature_list(pp: pd.DataFrame) -> list[str]:
    return ([c for c in _HAZARD_FEATURES if c in pp.columns]
            + [f"period_{p}" for p in range(N_PERIODS)])


# --------------------------------------------------------------------------- main
def main(ext: Extracts | None = None) -> dict:
    config.ensure_dirs()
    ext = ext or load_extracts(verbose=False)

    stale = config.ARTIFACTS / "phase4_multi_horizon_risk.csv"
    if len(ext.wo) == 0 or not (ext.wo["wo_type"] == "CM").any():
        stale.unlink(missing_ok=True)          # don't let Phase 5 read a stale run
        print("Phase 4 - Prognostics: SKIPPED (no corrective work orders to fit survival on)")
        return {"status": "skipped_no_work_orders"}

    labelled = add_labels(ext.sos, ext.wo, CENSOR)
    cutoff = labelled["sample_date"].quantile(config.PEER_TRAIN_FRACTION)
    peer = fit_peer_baseline(labelled, cutoff)
    full = add_features(labelled, peer)
    full["machine_frailty"] = machine_frailty(full, ext.wo).values

    data_cutoff = ext.wo["open_date"].max()
    fit_pool = full[full["sample_date"] <= data_cutoff].copy()

    train, test = model.time_split(fit_pool)
    pp_train, _, _ = to_person_period(train, data_cutoff)
    feats = _feature_list(pp_train)

    # NOTE: no class_weight balancing -- the hazard has to stay calibrated to the
    # true per-interval failure rate or the risk cards read 100% for everything.
    haz = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.4)),
    ]).fit(pp_train[feats], pp_train["event"])

    risk_all = predict_risk(haz, full, feats)

    # held-out check: does 30-day survival risk rank the actual 30-day outcomes?
    test_risk = risk_all.loc[test.index]
    y30 = ((test["days_to_next_cm"] > 0) & (test["days_to_next_cm"] <= 30)).astype(int)
    check = model.evaluate("survival risk_30d (held-out)", y30, test_risk["risk_30d"])

    km = kaplan_meier(train, data_cutoff)
    km.to_csv(config.ARTIFACTS / "phase4_km_baseline.csv", index=False)

    latest = latest_scored(full, risk_all)
    latest.to_csv(config.ARTIFACTS / "phase4_multi_horizon_risk.csv", index=False)

    n_events = int(pp_train["event"].sum())
    lines = ["Phase 4 - Survival Prognostics (discrete-time pooled logistic hazard)",
             "=" * 68, "",
             f"Administrative censoring horizon : {CENSOR} days ({N_PERIODS} x {STEP}d periods)",
             f"Person-period training rows      : {len(pp_train):,}  ({n_events} events)",
             f"Hazard features                  : {len(feats)}",
             f"Time split cutoff                : {pd.Timestamp(cutoff).date()}", "",
             "Held-out ranking check:", model.fmt_eval(check), "",
             f"KM baseline survival @30/60/90d  : "
             f"{km_at(km, 30):.3f} / {km_at(km, 60):.3f} / {km_at(km, 90):.3f}"]
    (config.ARTIFACTS / "phase4_metrics.txt").write_text("\n".join(lines) + "\n")

    from . import report
    report.phase4_summary(
        n_rows=len(pp_train),
        n_events=n_events,
        prauc_30d=float(check['pr_auc']),
        lift_30d=float(check.get('lift_over_base', check['pr_auc'] / max(check['base_rate'], 0.001))),
        km_30=km_at(km, 30),
        km_60=km_at(km, 60),
        km_90=km_at(km, 90),
        n_scored=len(latest),
        top_rows=latest.head(5).to_dict(orient='records'),
    )
    return {"held_out": check, "scored": len(latest), "events": n_events}


# ------------------------------------------------------------------------- helpers
def predict_risk(haz, df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Per-sample cumulative failure risk at each survival horizon."""
    base = df.copy()
    out = pd.DataFrame(index=df.index)
    surv = np.ones(len(df))
    horizons = {h: None for h in config.SURVIVAL_HORIZONS}
    for p in range(N_PERIODS):
        block = base.copy()
        for q in range(N_PERIODS):
            block[f"period_{q}"] = int(q == p)
        cols = [c for c in feats]
        h_p = haz.predict_proba(block[cols])[:, 1]
        surv = surv * (1.0 - h_p)
        reached = (p + 1) * STEP
        for h in config.SURVIVAL_HORIZONS:
            if reached == h or (horizons[h] is None and reached >= h):
                horizons[h] = 1.0 - surv
    for h in config.SURVIVAL_HORIZONS:
        out[f"risk_{h}d"] = np.clip(horizons[h], 0.001, 0.999).round(4)
    return out


def latest_scored(full: pd.DataFrame, risk_all: pd.DataFrame) -> pd.DataFrame:
    j = full.join(risk_all)
    latest = (j.sort_values("sample_date")
              .groupby(["machine_id", "component"], as_index=False).tail(1))
    cols = ["machine_id", "model", "model_family", "component", "component_family",
            "position", "sample_date", "smu_hours", "oil_hours", "lab_severity",
            "machine_frailty", "risk_30d", "risk_60d", "risk_90d"]
    return (latest[cols].sort_values("risk_30d", ascending=False).reset_index(drop=True))


def kaplan_meier(df: pd.DataFrame, data_cutoff: pd.Timestamp) -> pd.DataFrame:
    days_to_cutoff = (data_cutoff - df["sample_date"]).dt.days.clip(lower=0)
    censor_time = np.minimum(days_to_cutoff, CENSOR).to_numpy()
    dnext = df["days_to_next_cm"].to_numpy()
    observed = np.isfinite(dnext) & (dnext > 0) & (dnext <= censor_time)
    t = np.where(observed, dnext, censor_time)
    grid = np.arange(STEP, CENSOR + STEP, STEP)
    surv, s = [], 1.0
    for g in grid:
        at_risk = (t >= g - STEP).sum()
        events = (observed & (t > g - STEP) & (t <= g)).sum()
        if at_risk > 0:
            s *= (1 - events / at_risk)
        surv.append({"day": int(g), "survival": round(float(s), 4),
                     "at_risk": int(at_risk), "events": int(events)})
    return pd.DataFrame(surv)


def km_at(km: pd.DataFrame, day: int) -> float:
    row = km[km["day"] == day]
    return float(row["survival"].iloc[0]) if len(row) else float("nan")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
