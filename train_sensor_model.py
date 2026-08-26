"""
Train multi-target machine learning models on hydraulic sensor data.

Predicts 4 critical component health conditions:
  1. Cooler condition (%)
  2. Valve condition (%)
  3. Internal pump leakage (0, 1, 2)
  4. Hydraulic accumulator pressure (bar)

Run:  python train_sensor_model.py
Out:  artifacts/sensor_models.joblib, artifacts/sensor_metrics.txt
"""

from __future__ import annotations

import os
import warnings
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

TARGET_COLS = [
    "cooler_condition",
    "valve_condition",
    "pump_leakage",
    "accumulator_pressure"
]

TARGET_LABELS = {
    "cooler_condition": {100: "Full Efficiency (100%)", 20: "Reduced Efficiency (20%)", 3: "Close to Failure (3%)"},
    "valve_condition": {100: "Optimal (100%)", 90: "Small Lag (90%)", 80: "Severe Lag (80%)", 73: "Close to Failure (73%)"},
    "pump_leakage": {0: "No Leakage", 1: "Weak Leakage", 2: "Severe Leakage"},
    "accumulator_pressure": {130: "Optimal (130 bar)", 115: "Slightly Reduced (115 bar)", 100: "Severely Reduced (100 bar)", 90: "Close to Failure (90 bar)"}
}


def main():
    table_path = "data/processed/sensor_model_table.csv"
    if not os.path.exists(table_path):
        raise FileNotFoundError("Run 'python build_sensor_dataset.py' first to generate sensor_model_table.csv")

    df = pd.read_csv(table_path)
    feature_cols = [c for c in df.columns if c not in TARGET_COLS and c != "stable_flag"]

    X = df[feature_cols]

    # 80/20 train/test split
    train_idx, test_idx = train_test_split(df.index, test_size=0.20, random_state=42, stratify=df["cooler_condition"])
    
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]

    models = {}
    lines = ["==================================================",
             "HYDRAULIC SYSTEM COMPONENT CONDITION MONITORING",
             "==================================================\n"]

    print("\nTraining models for hydraulic system components...\n")

    for target in TARGET_COLS:
        y = df[target]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        # Use Random Forest Classifier for robust multiclass classification
        clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        models[target] = clf

        header = f"Target: {target.upper().replace('_', ' ')}"
        print(f"{header}")
        print("-" * len(header))
        print(f"  Accuracy: {acc * 100:.2f}% | Macro F1: {f1_macro:.4f}")
        
        rep = classification_report(y_test, y_pred, digits=3)
        print("  Classification Report:")
        for rline in rep.split("\n"):
            if rline.strip():
                print(f"    {rline}")
        print()

        lines.append(f"{header}")
        lines.append(f"Accuracy: {acc * 100:.2f}% | Macro F1: {f1_macro:.4f}")
        lines.append("Classification Report:\n" + rep + "\n")

        # Top features per target
        fi = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        top5 = ", ".join([f"{k} ({v:.3f})" for k, v in fi.head(5).items()])
        lines.append(f"Top 5 Drivers: {top5}\n" + "-"*50 + "\n")

    os.makedirs("artifacts", exist_ok=True)
    dump({"models": models, "features": feature_cols}, "artifacts/sensor_models.joblib")

    with open("artifacts/sensor_metrics.txt", "w") as f:
        f.write("\n".join(lines))

    print("Saved trained sensor models to 'artifacts/sensor_models.joblib'")
    print("Saved metrics evaluation to 'artifacts/sensor_metrics.txt'")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
