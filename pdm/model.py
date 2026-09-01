"""
Shared modelling helpers -- the time-split, the evaluation protocol and the
threshold rule that Phases 3-5 all use, so they can never disagree on what
"good" means.

Evaluation protocol (decided once, here):
  * split by TIME, never at random -- a random split lets the future teach the past
  * positives are ~3% of rows, so judge on PR-AUC and precision-at-capacity,
    never accuracy
  * always score the lab's own severity flag as a baseline. If the model cannot
    beat it on the held-out window, the honest answer is to ship the rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)

from . import config


# --------------------------------------------------------------------- splitting
def time_split(df: pd.DataFrame, frac: float = config.TIME_SPLIT_FRACTION,
               date_col: str = "sample_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earliest `frac` of the timeline trains; the rest tests. Falls back to a
    positional split when a date split would leave either side empty."""
    d = df.sort_values(date_col)
    cutoff = d[date_col].quantile(frac)
    train, test = d[d[date_col] < cutoff], d[d[date_col] >= cutoff]
    if len(train) < 10 or len(test) < 5 or train["label"].nunique() < 2:
        k = max(1, int(len(d) * frac))
        train, test = d.iloc[:k], d.iloc[k:] if k < len(d) else d.iloc[:k]
    return train.copy(), test.copy()


# -------------------------------------------------------------------- evaluation
def precision_at_capacity(y_true, scores, frac: float = config.INSPECTION_CAPACITY_FRAC):
    """Of the riskiest `frac` of samples an engineer actually has time to chase,
    what share are real -- and what share of all real problems does that catch."""
    y = np.asarray(y_true)
    n = max(1, int(round(len(scores) * frac)))
    order = np.argsort(-np.asarray(scores))[:n]
    caught = y[order].sum()
    return caught / n, caught / max(1, y.sum())


def evaluate(name: str, y_true, scores, threshold: float | None = None) -> dict:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    base = float(y.mean())
    single_class = len(np.unique(y)) < 2
    ap = base if single_class else float(average_precision_score(y, s))
    auc = float("nan") if single_class else float(roc_auc_score(y, s))
    p_cap, r_cap = precision_at_capacity(y, s)
    row = {
        "model": name,
        "pr_auc": round(ap, 4),
        "base_rate": round(base, 4),
        "lift_over_base": round(ap / base, 2) if base > 0 else float("nan"),
        "roc_auc": round(auc, 4),
        "precision_at_capacity": round(float(p_cap), 3),
        "recall_at_capacity": round(float(r_cap), 3),
        "brier": round(float(brier_score_loss(y, np.clip(s, 0, 1))), 4) if not single_class else 0.0,
    }
    if threshold is not None:
        pred = (s >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        row["threshold"] = round(float(threshold), 4)
        row["alerts"] = int(pred.sum())
        row["precision_at_threshold"] = round(tp / max(1, pred.sum()), 3)
        row["recall_at_threshold"] = round(tp / max(1, y.sum()), 3)
    return row


def pick_threshold(y_true, scores, min_precision: float = 0.30) -> float:
    """Lowest score threshold that still holds precision >= min_precision -- i.e.
    the most recall you can trust. Falls back to best-F1 when nothing qualifies."""
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return 0.5
    p, r, t = precision_recall_curve(y, scores)
    p, r = p[:-1], r[:-1]
    ok = np.where(p >= min_precision)[0]
    if len(ok):
        return float(t[ok[int(np.argmax(r[ok]))]])
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[int(np.argmax(f1))])


def fmt_eval(row: dict) -> str:
    s = (f"{row['model']:<28} PR-AUC {row['pr_auc']:.3f}  "
         f"(base {row['base_rate']:.3f}, lift {row['lift_over_base']}x)  "
         f"ROC-AUC {row['roc_auc']:.3f}  "
         f"prec@cap {row['precision_at_capacity']:.2f}  rec@cap {row['recall_at_capacity']:.2f}")
    if "threshold" in row:
        s += (f"\n{'':<28} @thr {row['threshold']:.3f}: prec {row['precision_at_threshold']:.2f}  "
              f"rec {row['recall_at_threshold']:.2f}  alerts {row['alerts']}")
    return s
