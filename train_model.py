"""
Train and honestly evaluate the 30-day corrective-maintenance risk model.

Rules that matter more than the algorithm choice:
  * split by TIME, never randomly (a random split lets the future teach the past)
  * compare against the lab's own severity flag -- if you can't beat it, ship it
  * judge on PR-AUC and precision@top-k, not accuracy (positives are ~3% of rows)

Run:  python train_model.py
Out:  artifacts/model.joblib, artifacts/metrics.txt, artifacts/feature_importance.csv
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_dataset import TRAIN_FRACTION, feature_columns

TOP_K = 0.10          # inspection capacity: you can chase the riskiest 10% of samples


def precision_at_k(y_true, scores, k=TOP_K):
    n = max(1, int(round(len(scores) * k)))
    order = np.argsort(-scores)[:n]
    caught = y_true.values[order].sum()
    return caught / n, caught / max(1, y_true.sum())


def evaluate(name, y, scores, thr=None, lines=None):
    ap = average_precision_score(y, scores)
    auc = roc_auc_score(y, scores)
    p_at_k, r_at_k = precision_at_k(y, scores)
    base = y.mean()
    txt = (f"{name:<26} PR-AUC {ap:.3f} (baseline {base:.3f}, lift {ap/base:.1f}x) | "
           f"ROC-AUC {auc:.3f} | top-{int(TOP_K*100)}%: precision {p_at_k:.2f}, recall {r_at_k:.2f}")
    if thr is not None:
        pred = (scores >= thr).astype(int)
        txt += (f"\n{'':<26} @thr={thr:.3f}: precision {precision_score(y, pred, zero_division=0):.2f}, "
                f"recall {recall_score(y, pred, zero_division=0):.2f}, alerts {pred.sum()}/{len(pred)}")
    print(txt)
    if lines is not None:
        lines.append(txt)
    return ap


def pick_threshold(y, scores, min_precision=0.30):
    """Lowest threshold that still keeps precision acceptable -> max recall you can trust."""
    p, r, t = precision_recall_curve(y, scores)
    ok = np.where(p[:-1] >= min_precision)[0]
    if len(ok) == 0:
        f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
        return float(t[int(np.argmax(f1))])
    return float(t[ok[int(np.argmax(r[:-1][ok]))]])


def main():
    df = pd.read_csv("data/processed/model_table.csv", parse_dates=["sample_date"])
    df = df[~df.censored & ~df.post_repair].copy()

    cutoff = df["sample_date"].quantile(TRAIN_FRACTION)
    train, test = df[df.sample_date < cutoff], df[df.sample_date >= cutoff]
    feats = [c for c in feature_columns(df) if train[c].notna().any()]  # drop all-empty columns
    Xtr, ytr = train[feats], train["label"]
    Xte, yte = test[feats], test["label"]

    lines = [f"train: {len(train):,} samples to {pd.Timestamp(cutoff).date()} "
             f"(positives {int(ytr.sum())}, {ytr.mean():.2%})",
             f"test : {len(test):,} samples after  {pd.Timestamp(cutoff).date()} "
             f"(positives {int(yte.sum())}, {yte.mean():.2%})", ""]
    print("\n".join(lines))

    # 0. Rule baseline -- what the lab flag alone would have told you.
    rule = test["lab_severity_num"].values.astype(float)
    evaluate("rule: lab severity", yte, rule, lines=lines)

    # 1. Explainable linear baseline.
    logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3)),
    ]).fit(Xtr, ytr)
    s_lr = logreg.predict_proba(Xte)[:, 1]
    ap_lr = evaluate("logistic regression", yte, s_lr, lines=lines)

    # 2. Gradient boosting -- handles NaNs and non-linear thresholds natively.
    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=15,
        min_samples_leaf=25, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.15,
        random_state=0).fit(Xtr, ytr)
    s_hgb = hgb.predict_proba(Xte)[:, 1]
    ap_hgb = evaluate("gradient boosting", yte, s_hgb,
                      thr=pick_threshold(ytr, hgb.predict_proba(Xtr)[:, 1]), lines=lines)

    best_name, best, scores = (("gradient boosting", hgb, s_hgb) if ap_hgb >= ap_lr
                               else ("logistic regression", logreg, s_lr))
    # alert threshold comes from the SELECTED model, fitted on training scores only
    thr = pick_threshold(ytr, best.predict_proba(Xtr)[:, 1])
    evaluate(f"selected on test", yte, scores, thr=thr, lines=lines)
    lines.append(f"\nselected: {best_name} | Brier {brier_score_loss(yte, scores):.4f}")
    print(lines[-1])

    # Which signals actually drive it (permutation on the held-out window).
    imp = permutation_importance(best, Xte, yte, n_repeats=5, random_state=0,
                                 scoring="average_precision", n_jobs=-1)
    fi = (pd.DataFrame({"feature": feats, "importance": imp.importances_mean})
          .sort_values("importance", ascending=False))
    fi.to_csv("artifacts/feature_importance.csv", index=False)
    lines.append("\ntop 15 features:\n" + fi.head(15).to_string(index=False))
    print(lines[-1])

    dump({"model": best, "features": feats, "threshold": thr, "name": best_name},
         "artifacts/model.joblib")
    with open("artifacts/metrics.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open("artifacts/run_config.json", "w") as f:
        json.dump({"cutoff": str(pd.Timestamp(cutoff).date()), "threshold": thr,
                   "model": best_name, "n_features": len(feats)}, f, indent=2)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.makedirs("artifacts", exist_ok=True)
    main()
