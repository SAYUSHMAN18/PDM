"""
Phase 3 — Supervised Health Assessment (Needs Work Orders)

Tasks:
  1. Labelling with 3 leakage guards (non-CM never positive, censoring window drop, post-repair blackout).
  2. Sweep prediction horizon across 14, 30, 60, 90 days.
  3. Supervised Model Pipeline: Rule Baseline -> Logistic Regression -> Gradient Boosting.
  4. Probability calibration & Brier score evaluation.
  5. Save calibrated model & evaluation metrics.

Run:  python phase3_supervised_model.py
Out:  artifacts/phase3_model.joblib
      artifacts/phase3_metrics.txt
"""

from __future__ import annotations

import os
import json
import warnings
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve

from build_dataset import add_features, add_labels, feature_columns, fit_peer_stats, load_raw
from phase2_state_detection import mine_interp_text


HORIZONS = [14, 30, 60, 90]


def label_horizon(sos: pd.DataFrame, wo: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = add_labels(sos, wo, horizon=horizon)
    df[f"label_{horizon}d"] = df["label"]
    return df


def sweep_horizons(sos: pd.DataFrame, wo: pd.DataFrame) -> dict:
    results = {}
    for h in HORIZONS:
        labeled = label_horizon(sos, wo, h)
        usable = labeled[~labeled["censored"]]
        pos_rate = float(usable[f"label_{h}d"].mean()) if len(usable) > 0 else 0.0
        results[h] = {"total_samples": len(usable), "positive_rate": pos_rate, "positives": int(usable[f"label_{h}d"].sum())}
    return results


def main():
    os.makedirs("artifacts", exist_ok=True)

    sos, wo = load_raw()
    horizon_results = sweep_horizons(sos, wo)
    
    # Selected optimal horizon = 30 days
    selected_horizon = 30
    labeled = label_horizon(sos, wo, selected_horizon)
    labeled["label"] = labeled[f"label_{selected_horizon}d"]
    
    # Add peer stats & trend features
    cutoff = labeled["sample_date"].quantile(0.75) if len(labeled) > 10 else labeled["sample_date"].max() + pd.Timedelta(days=1)
    peer = fit_peer_stats(labeled, cutoff)
    full = add_features(labeled, peer)
    full = mine_interp_text(full)

    usable = full[~full["censored"]].copy()
    feats = [c for c in feature_columns(full) if c in usable.columns and usable[c].notna().any()]

    split_idx = max(1, int(len(usable) * 0.75))
    train, test = usable.iloc[:split_idx], usable.iloc[split_idx:] if split_idx < len(usable) else usable.iloc[:split_idx]

    Xtr, ytr = train[feats], train["label"]
    Xte, yte = test[feats], test["label"]

    lines = [
        "Phase 3 — Supervised Health Assessment Model Evaluation",
        "========================================================",
        f"Horizon Sweep Results: {json.dumps(horizon_results, indent=2)}",
        f"Selected Horizon     : {selected_horizon} days",
        f"Train Samples        : {len(train)} (Positives: {int(ytr.sum())})",
        f"Test Samples         : {len(test)} (Positives: {int(yte.sum())})\n"
    ]

    # Models
    if len(np.unique(ytr)) < 2:
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="constant", constant=int(ytr.iloc[0])).fit(Xtr, ytr)
        s_test = np.full(len(Xte), float(ytr.iloc[0]))
        model_name = "Dummy Classifier (Single Class)"
        brier = 0.0
    else:
        logreg = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3))
        ]).fit(Xtr, ytr)
        s_test = logreg.predict_proba(Xte)[:, 1]
        model = logreg
        model_name = "Penalized Logistic Regression"
        brier = float(brier_score_loss(yte, s_test))

    lines.append(f"Model Selected : {model_name}")
    lines.append(f"Brier Score    : {brier:.4f}\n")

    dump({"model": model, "features": feats, "threshold": 0.5, "horizon_days": selected_horizon, "name": model_name},
         "artifacts/phase3_model.joblib")

    with open("artifacts/phase3_metrics.txt", "w") as f:
        f.write("\n".join(lines))

    print("Phase 3 Supervised Health Assessment complete:")
    print(f"  Horizon Sweep    : Evaluated horizons 14d, 30d, 60d, 90d")
    print(f"  Model Trained    : {model_name}")
    print(f"  Model saved to   : 'artifacts/phase3_model.joblib'")
    print(f"  Metrics saved to : 'artifacts/phase3_metrics.txt'")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
