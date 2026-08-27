"""
Phase 3 - Health Assessment: the supervised model  (weeks 10-18, needs work orders)

A calibrated probability that a corrective work order lands on this machine +
compartment within the horizon, that beats the Phase 2 rule baseline.

  * Horizon sweep      label at 14 / 30 / 60 / 90 days and report how the
                       positive count and base rate move -- the horizon is a real
                       hyper-parameter, not a given.
  * Leakage guards     from features.add_labels: PM work orders are never
                       positives, the trailing horizon window is censored out,
                       and post-repair samples are dropped.
  * Models in order    rule baseline (lab severity)  ->  penalised logistic
                       ->  gradient boosting with monotonic constraints
                       (rising iron must never lower predicted risk).
  * Calibration        isotonic / sigmoid on a time-aware inner split.
  * Evaluation         time split, PR-AUC + precision-at-capacity + Brier.
                       Never accuracy.

Outputs
  artifacts/phase3_model.joblib      calibrated model + feature list + threshold
  artifacts/phase3_metrics.txt       the full comparison table
  artifacts/feature_importance.csv   permutation importance on the held-out window
  data/processed/risk_scores.csv     latest sample per machine-compartment, scored
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, model
from .data import Extracts, load_extracts
from .features import (add_features, add_labels, feature_columns,
                       fit_peer_baseline, monotone_constraints)

MODEL_PATH = config.ARTIFACTS / "phase3_model.joblib"


# --------------------------------------------------------------------- horizon sweep
def horizon_sweep(ext: Extracts) -> pd.DataFrame:
    rows = []
    for h in config.HORIZONS:
        lab = add_labels(ext.sos, ext.wo, h)
        usable = lab[lab["usable"]]
        rows.append({
            "horizon_days": h,
            "usable_samples": int(len(usable)),
            "positives": int(usable["label"].sum()),
            "base_rate": round(float(usable["label"].mean()) if len(usable) else 0.0, 4),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ model training
def _build_models(Xtr, ytr, feats):
    """Return {name: fitted estimator} for every candidate that can train here."""
    out = {}
    if ytr.nunique() < 2:
        out["constant"] = DummyClassifier(strategy="constant",
                                          constant=int(ytr.iloc[0])).fit(Xtr, ytr)
        return out

    out["logistic"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3)),
    ]).fit(Xtr, ytr)

    out["gradient_boosting"] = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.07, max_leaf_nodes=15,
        min_samples_leaf=min(25, max(2, len(Xtr) // 4)),
        l2_regularization=1.0, class_weight="balanced",
        monotonic_cst=monotone_constraints(feats),
        early_stopping=False, random_state=0,
    ).fit(Xtr, ytr)
    return out


def _calibrate(est, X_cal, y_cal):
    """Calibrate an already-fitted estimator on a held-out (later) time slice."""
    pos = int(y_cal.sum())
    if y_cal.nunique() < 2 or pos < 10:
        return est                                   # not enough events to calibrate
    method = "isotonic" if pos >= 40 else "sigmoid"
    try:
        return CalibratedClassifierCV(est, method=method, cv="prefit").fit(X_cal, y_cal)
    except Exception:
        return est


def _proba(est, X) -> np.ndarray:
    """P(label == 1), robust to a degenerate single-class estimator."""
    p = np.asarray(est.predict_proba(X))
    if p.ndim == 2 and p.shape[1] > 1:
        return p[:, 1]
    classes = getattr(est, "classes_", [0])
    only = classes[0] if len(classes) else 0
    col = np.ravel(p)
    return col if only == 1 else 1.0 - col


# --------------------------------------------------------------------------- main
def main(ext: Extracts | None = None) -> dict:
    config.ensure_dirs()
    ext = ext or load_extracts(verbose=False)

    sweep = horizon_sweep(ext)
    h = config.PRIMARY_HORIZON

    labelled = add_labels(ext.sos, ext.wo, h)
    cutoff = labelled["sample_date"].quantile(config.PEER_TRAIN_FRACTION)
    peer = fit_peer_baseline(labelled, cutoff)
    full = add_features(labelled, peer)

    usable = full[full["usable"]].copy()
    feats = feature_columns(usable)
    train, test = model.time_split(usable)

    # inner time split of the training rows: fit on the earlier part, calibrate
    # probabilities on the later part (never on the test window)
    k = max(1, int(len(train) * 0.85))
    fit_df, cal_df = train.iloc[:k], train.iloc[k:]
    if len(cal_df) < 20 or cal_df["label"].nunique() < 2:
        fit_df, cal_df = train, train
    Xfit, yfit = fit_df[feats], fit_df["label"]
    Xcal, ycal = cal_df[feats], cal_df["label"]
    Xtr, ytr = train[feats], train["label"]
    Xte, yte = test[feats], test["label"]

    results = [model.evaluate("rule: lab severity", yte, test["lab_severity_num"])]
    fitted = _build_models(Xfit, yfit, feats)

    scored = {}
    for name, est in fitted.items():
        cal = _calibrate(est, Xcal, ycal)
        s = _proba(cal, Xte)
        scored[name] = (cal, s)
        results.append(model.evaluate(name, yte, s))

    # pick the best model by held-out PR-AUC; fall back to the rule if nothing beats it
    ml = [r for r in results if r["model"] != "rule: lab severity"]
    best_row = max(ml, key=lambda r: r["pr_auc"]) if ml else results[0]
    rule_row = results[0]
    ship_rule = best_row["pr_auc"] <= rule_row["pr_auc"] and ytr.nunique() > 1

    best_name = best_row["model"]
    best_est, best_scores = scored[best_name]
    thr = model.pick_threshold(ytr, _proba(best_est, Xtr))
    final_row = model.evaluate(f"SELECTED: {best_name}", yte, best_scores, threshold=thr)
    results.append(final_row)

    # permutation importance on the held-out window (what actually drives the score)
    if ytr.nunique() > 1 and yte.nunique() > 1:
        sub = Xte if len(Xte) <= 2000 else Xte.sample(2000, random_state=0)
        imp = permutation_importance(best_est, sub, yte.loc[sub.index], n_repeats=3,
                                     random_state=0, scoring="average_precision")
        importances = imp.importances_mean
    else:
        importances = Xte.std(axis=0).fillna(0).to_numpy()
    fi = (pd.DataFrame({"feature": feats, "importance": importances})
          .sort_values("importance", ascending=False).reset_index(drop=True))
    fi.to_csv(config.ARTIFACTS / "feature_importance.csv", index=False)

    bundle = {
        "model": best_est, "features": feats, "threshold": float(thr),
        "horizon_days": h, "name": best_name, "peer_baseline": peer,
        "trained_through": str(pd.Timestamp(cutoff).date()),
        "ship_rule_instead": bool(ship_rule),
    }
    dump(bundle, MODEL_PATH)

    lines = ["Phase 3 - Supervised Health Assessment", "=" * 60, "",
             "Horizon sweep:", sweep.to_string(index=False), "",
             f"Primary horizon      : {h} days",
             f"Train / test samples : {len(train):,} ({int(ytr.sum())} pos) / "
             f"{len(test):,} ({int(yte.sum())} pos)",
             f"Features             : {len(feats)}",
             f"Time split cutoff    : {pd.Timestamp(cutoff).date()}", "", "Model comparison:"]
    lines += [model.fmt_eval(r) for r in results]
    if ship_rule:
        lines += ["", "*** No model beat the lab-severity rule on the held-out window. "
                  "Ship the Phase 2 rule; revisit Phase 3 with more events. ***"]
    (config.ARTIFACTS / "phase3_metrics.txt").write_text("\n".join(lines) + "\n")
    with open(config.ARTIFACTS / "phase3_horizon_sweep.json", "w") as f:
        json.dump(sweep.to_dict(orient="records"), f, indent=2)

    risk = score_latest(full, bundle)
    risk.to_csv(config.DATA_PROCESSED / "risk_scores.csv", index=False)

    print("Phase 3 - Supervised Health Assessment")
    print(f"  horizon sweep         : " + ", ".join(
        f"{r.horizon_days}d={r.positives}pos" for r in sweep.itertuples()))
    print(f"  selected model        : {best_name}"
          + ("  (but rule baseline wins -- ship the rule)" if ship_rule else ""))
    print(f"  held-out PR-AUC       : {best_row['pr_auc']:.3f} "
          f"(rule {rule_row['pr_auc']:.3f}, base {best_row['base_rate']:.3f})")
    print(f"  precision @ capacity  : {final_row['precision_at_capacity']:.2f}  "
          f"(recall {final_row['recall_at_capacity']:.2f})")
    print(f"  model saved           : artifacts/phase3_model.joblib")
    return {"sweep": sweep.to_dict(orient="records"), "results": results,
            "ship_rule": ship_rule, "selected": best_name}


# ----------------------------------------------------------------- scoring helper
def score_latest(full: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    latest = (full.sort_values("sample_date")
              .groupby(["machine_id", "component"], as_index=False).tail(1)
              .reset_index(drop=True))
    feats = bundle["features"]
    latest["risk"] = _proba(bundle["model"], latest[feats])
    thr = bundle["threshold"]
    latest["risk_level"] = pd.cut(
        latest["risk"], [-1, max(0.01, thr * 0.5), thr, 0.75, 1.1],
        labels=["LOW", "WATCH", "HIGH", "CRITICAL"], duplicates="drop")
    cols = ["machine_id", "model", "model_family", "component", "component_family",
            "position", "sample_date", "smu_hours", "oil_hours", "lab_severity",
            "risk", "risk_level"]
    return (latest[cols].sort_values("risk", ascending=False).reset_index(drop=True))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
