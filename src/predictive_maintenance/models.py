from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "corrective_wo_within_horizon"
EXCLUDED = {
    TARGET,
    "sample_number",
    "asset_id",
    "sample_date",
    "horizon_days",
}


@dataclass
class TrainedModel:
    name: str
    pipeline: Any
    features: list[str]
    metrics: dict[str, Any]
    cutoff_date: pd.Timestamp
    explain_pipeline: Pipeline | None = None

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[self.features])[:, 1]

    def get_feature_importances(self) -> dict[str, float]:
        """Extract normalized feature importances or absolute model coefficients."""
        try:
            source = self.explain_pipeline or self.pipeline
            model = source.named_steps["model"]
            preprocess = source.named_steps["preprocess"]

            if hasattr(preprocess, "get_feature_names_out"):
                raw_names = preprocess.get_feature_names_out()
            else:
                raw_names = [f"feature_{i}" for i in range(len(self.features))]

            if hasattr(model, "feature_importances_"):
                scores = model.feature_importances_
            elif hasattr(model, "coef_"):
                scores = np.abs(model.coef_[0])
            else:
                return {}

            clean_scores: dict[str, float] = {}
            for name, score in zip(raw_names, scores):
                clean_name = name.split("__")[-1]
                clean_scores[clean_name] = clean_scores.get(clean_name, 0.0) + float(score)

            total = sum(clean_scores.values())
            if total > 0:
                clean_scores = {k: round(v / total, 4) for k, v in clean_scores.items()}
            return dict(sorted(clean_scores.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            return {}


def _safe_metrics(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, Any] = {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "precision_at_0_5": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "test_rows": int(len(y_true)),
        "test_positive_rate": float(y_true.mean()),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities)) if y_true.nunique() == 2 else None
    )
    return metrics


def train_failure_models(training: pd.DataFrame) -> tuple[TrainedModel, pd.DataFrame]:
    """Train and sigmoid-calibrate baselines with chronological train/calibration/test blocks."""
    if len(training) < 60:
        raise ValueError("At least 60 labelled samples are required for chronological calibration and testing.")
    if training[TARGET].nunique() < 2:
        raise ValueError("Training data must contain both positive and negative outcomes.")

    data = training.sort_values("sample_date").reset_index(drop=True)
    split_candidates: list[tuple[int, int]] = []
    for train_end in range(int(len(data) * 0.55), int(len(data) * 0.71)):
        for calibration_end in range(int(len(data) * 0.75), int(len(data) * 0.86)):
            if calibration_end <= train_end:
                continue
            blocks = [data.iloc[:train_end], data.iloc[train_end:calibration_end], data.iloc[calibration_end:]]
            if all(len(block) >= 8 and block[TARGET].nunique() == 2 for block in blocks):
                split_candidates.append((train_end, calibration_end))
    if not split_candidates:
        raise ValueError(
            "Chronological train, calibration, and test blocks must each contain positive and negative outcomes."
        )
    train_end, calibration_end = split_candidates[len(split_candidates) // 2]
    train = data.iloc[:train_end].copy()
    calibration = data.iloc[train_end:calibration_end].copy()
    test = data.iloc[calibration_end:].copy()

    features = [column for column in data.columns if column not in EXCLUDED]
    categorical = [column for column in features if data[column].dtype == "object" or str(data[column].dtype).startswith("string")]
    numeric = [column for column in features if column not in categorical]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
    }
    rows = []
    fitted: dict[str, Any] = {}
    explainers: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        base_pipeline = Pipeline([("preprocess", clone(preprocess)), ("model", estimator)])
        base_pipeline.fit(train[features], train[TARGET])
        try:
            from sklearn.frozen import FrozenEstimator
            calibrated = CalibratedClassifierCV(FrozenEstimator(base_pipeline), method="sigmoid")
        except ImportError:
            calibrated = CalibratedClassifierCV(base_pipeline, method="sigmoid", cv="prefit")
        calibrated.fit(calibration[features], calibration[TARGET])
        probabilities = calibrated.predict_proba(test[features])[:, 1]
        metrics = _safe_metrics(test[TARGET], probabilities)
        metrics["model"] = name
        metrics["train_rows"] = int(len(train))
        metrics["calibration_rows"] = int(len(calibration))
        rows.append(metrics)
        fitted[name] = calibrated
        explainers[name] = base_pipeline

    leaderboard = pd.DataFrame(rows).sort_values(
        ["average_precision", "brier_score"], ascending=[False, True]
    )
    winner = str(leaderboard.iloc[0]["model"])
    best_metrics = next(row for row in rows if row["model"] == winner)
    model = TrainedModel(
        name=winner,
        pipeline=fitted[winner],
        features=features,
        metrics=best_metrics,
        cutoff_date=pd.Timestamp(test["sample_date"].min()),
        explain_pipeline=explainers[winner],
    )
    return model, leaderboard


class BayesianComponentBaseline:
    """Empirical-Bayes component event-rate baseline with credible intervals."""

    def __init__(self, prior_strength: float = 12.0):
        self.prior_strength = prior_strength
        self.table_: pd.DataFrame | None = None
        self.fleet_: dict[str, float] | None = None

    def fit(self, training: pd.DataFrame) -> "BayesianComponentBaseline":
        fleet_rate = float(training[TARGET].mean())
        alpha0 = max(0.5, fleet_rate * self.prior_strength)
        beta0 = max(0.5, (1 - fleet_rate) * self.prior_strength)
        grouped = training.groupby(["machine_model", "component"], dropna=False)[TARGET].agg(["sum", "count"])
        grouped["alpha"] = alpha0 + grouped["sum"]
        grouped["beta"] = beta0 + grouped["count"] - grouped["sum"]
        grouped["posterior_mean"] = grouped["alpha"] / (grouped["alpha"] + grouped["beta"])
        grouped["credible_low_90"] = beta.ppf(0.05, grouped["alpha"], grouped["beta"])
        grouped["credible_high_90"] = beta.ppf(0.95, grouped["alpha"], grouped["beta"])
        self.table_ = grouped.reset_index()
        self.fleet_ = {
            "alpha": alpha0 + training[TARGET].sum(),
            "beta": beta0 + len(training) - training[TARGET].sum(),
        }
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.table_ is None or self.fleet_ is None:
            raise RuntimeError("Fit the Bayesian baseline before prediction.")
        merged = frame[["machine_model", "component"]].merge(
            self.table_, on=["machine_model", "component"], how="left"
        )
        fleet_mean = self.fleet_["alpha"] / (self.fleet_["alpha"] + self.fleet_["beta"])
        merged["posterior_mean"] = merged["posterior_mean"].fillna(fleet_mean)
        merged["credible_low_90"] = merged["credible_low_90"].fillna(
            beta.ppf(0.05, self.fleet_["alpha"], self.fleet_["beta"])
        )
        merged["credible_high_90"] = merged["credible_high_90"].fillna(
            beta.ppf(0.95, self.fleet_["alpha"], self.fleet_["beta"])
        )
        return merged[["posterior_mean", "credible_low_90", "credible_high_90"]]


def score_telemetry_anomalies(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Fit an unsupervised snapshot anomaly model to cleaned telemetry."""
    out = telemetry.copy()
    features = [
        "gap_hours",
        "operating_hours_delta",
        "odometer_delta",
        "distance_per_operating_hour",
        "utilization_rate",
    ]
    usable = out[features].replace([np.inf, -np.inf], np.nan)
    if len(out) < 20 or usable.notna().sum().sum() == 0:
        out["telemetry_anomaly_score"] = np.nan
        out["telemetry_anomaly"] = False
        return out

    matrix = SimpleImputer(strategy="median", add_indicator=True).fit_transform(usable)
    model = IsolationForest(n_estimators=250, contamination=0.05, random_state=42, n_jobs=-1)
    labels = model.fit_predict(matrix)
    raw_score = -model.score_samples(matrix)
    low, high = np.nanmin(raw_score), np.nanmax(raw_score)
    normalized = (raw_score - low) / (high - low) if high > low else np.zeros_like(raw_score)
    out["telemetry_anomaly_score"] = normalized
    out["telemetry_anomaly"] = labels == -1
    return out
