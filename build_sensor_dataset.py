"""
Build tabular feature matrix from raw hydraulic sensor .txt files.

Sensors processed (17 total):
  - Pressure: PS1, PS2, PS3, PS4, PS5, PS6 (100 Hz, 6000 values/cycle)
  - Motor Power: EPS1 (100 Hz, 6000 values/cycle)
  - Volume Flow: FS1, FS2 (10 Hz, 600 values/cycle)
  - Temperature: TS1, TS2, TS3, TS4 (1 Hz, 60 values/cycle)
  - Vibration: VS1 (1 Hz, 60 values/cycle)
  - Efficiency: SE (1 Hz, 60 values/cycle)
  - Cooling Efficiency & Power: CE, CP (1 Hz, 60 values/cycle)

Targets from profile.txt:
  - cooler_condition (%, 100=full, 20=reduced, 3=close to failure)
  - valve_condition (%, 100=optimal, 90=small lag, 80=severe lag, 73=close to failure)
  - pump_leakage (0=none, 1=weak, 2=severe)
  - accumulator_pressure (bar, 130=optimal, 115=slightly reduced, 100=severely reduced, 90=close to failure)
  - stable_flag (0=stable, 1=unstable)

Run:  python build_sensor_dataset.py
Out:  data/processed/sensor_model_table.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

SENSOR_NAMES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1", "SE", "CE", "CP"
]

TARGET_COLS = [
    "cooler_condition",
    "valve_condition",
    "pump_leakage",
    "accumulator_pressure",
    "stable_flag"
]


def find_raw_file(filename: str) -> str:
    """Find file in data/raw or any subdirectory beneath it."""
    for root, _, files in os.walk("data/raw"):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"Could not find '{filename}' anywhere in data/raw/")


def extract_sensor_features(df_sensor: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Extract statistical summary metrics for each 60-second cycle."""
    arr = df_sensor.values
    feats = {}
    feats[f"{prefix}__mean"] = np.mean(arr, axis=1)
    feats[f"{prefix}__std"] = np.std(arr, axis=1)
    feats[f"{prefix}__min"] = np.min(arr, axis=1)
    feats[f"{prefix}__max"] = np.max(arr, axis=1)
    feats[f"{prefix}__median"] = np.median(arr, axis=1)
    feats[f"{prefix}__q25"] = np.quantile(arr, 0.25, axis=1)
    feats[f"{prefix}__q75"] = np.quantile(arr, 0.75, axis=1)
    feats[f"{prefix}__range"] = feats[f"{prefix}__max"] - feats[f"{prefix}__min"]
    return pd.DataFrame(feats)


def build():
    feature_dfs = []
    print("Loading sensor data and extracting cycle summary features...")
    
    for sname in SENSOR_NAMES:
        filepath = find_raw_file(f"{sname}.txt")
        print(f"  Processing {sname:<5} from {filepath}")
        sensor_raw = pd.read_csv(filepath, sep=r"\s+", header=None)
        feats = extract_sensor_features(sensor_raw, sname)
        feature_dfs.append(feats)

    X_all = pd.concat(feature_dfs, axis=1)

    profile_path = find_raw_file("profile.txt")
    print(f"Loading target profiles from {profile_path}")
    y_raw = pd.read_csv(profile_path, sep=r"\s+", header=None)
    y_raw.columns = TARGET_COLS

    full = pd.concat([X_all, y_raw], axis=1)
    return full


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    full_table = build()
    out_path = "data/processed/sensor_model_table.csv"
    full_table.to_csv(out_path, index=False)

    print(f"\nSuccessfully built sensor model table!")
    print(f"  Total cycles (rows): {len(full_table):,}")
    print(f"  Total features     : {len(full_table.columns) - len(TARGET_COLS)}")
    print(f"  Targets saved      : {', '.join(TARGET_COLS)}")
    print(f"  Output saved to    : {out_path}")
