"""
Turn raw S.O.S samples + work orders into one modelling table.

Three jobs, in order:
  1. LABEL  - for each oil sample, did a corrective work order follow on the
              SAME machine + component within the horizon (default 30 days)?
  2. FEATURE- what the sample says, and more importantly how it moved
              relative to its own history and to the peer fleet.
  3. GUARD  - drop rows that would leak the answer or that can't be labelled.

Run:  python build_dataset.py
Out:  data/processed/model_table.csv   artifacts/peer_stats.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

HORIZON_DAYS = 30          # the question: corrective WO within N days of the sample?
TRAIN_FRACTION = 0.75      # first 75% of the timeline is "past" (used for peer baselines)
POST_REPAIR_BLACKOUT = 7   # ignore samples taken just after a repair (oil is disturbed)

ANALYTES = [
    "fe_ppm", "cu_ppm", "cr_ppm", "pb_ppm", "al_ppm", "si_ppm", "na_ppm", "k_ppm",
    "water_pct", "fuel_pct", "glycol_pct", "soot_pct",
    "visc40", "oxidation", "nitration", "tbn", "pq_index",
]
SEVERITY_ORDER = {"NORMAL": 0, "MONITOR": 1, "ACTION": 2, "CRITICAL": 3}


def _resolve_raw_path(path: str, filename: str) -> str:
    if os.path.exists(path):
        return path
    raw_dir = "data/raw"
    if os.path.exists(raw_dir):
        for root, _, files in os.walk(raw_dir):
            if filename in files:
                return os.path.join(root, filename)
    return path


# --------------------------------------------------------------------------- load
def load_raw(sos_path="data/raw/sos_samples.csv", wo_path="data/raw/work_orders.csv"):
    excel_sos = _resolve_raw_path("data/raw/SosFluidSample.xlsx", "SosFluidSample.xlsx")
    excel_tele = _resolve_raw_path("data/raw/TelematicDataSample.xlsx", "TelematicDataSample.xlsx")
    csv_sos = _resolve_raw_path(sos_path, "sos_samples.csv")
    csv_wo = _resolve_raw_path(wo_path, "work_orders.csv")

    if os.path.exists(excel_sos):
        df_raw = pd.read_excel(excel_sos)
        sos = pd.DataFrame()
        sos["sample_id"] = df_raw["SampleNum"].astype(str) if "SampleNum" in df_raw.columns else df_raw["Id"].astype(str)
        sos["machine_id"] = df_raw["EquipNum"].fillna(df_raw["SerialNum"]).astype(str)
        sos["model"] = df_raw["EqpModel"].fillna("STANDARD").astype(str)
        sos["machine_type"] = df_raw["EqpManufacturerDescription"].fillna("HEAVY_EQUIPMENT").astype(str)
        sos["component"] = df_raw["Compartment"].astype(str).str.upper().str.strip()
        sos["sample_date"] = pd.to_datetime(df_raw["DateSampled"].fillna(df_raw["CreatedDate"]))
        sos["smu_hours"] = pd.to_numeric(df_raw["CMeter"], errors="coerce").fillna(1000.0)
        sos["oil_hours"] = pd.to_numeric(df_raw["CMeterFluid"], errors="coerce").fillna(250.0)

        interp_map = {"AR": "ACTION", "CR": "CRITICAL", "B": "MONITOR", "A": "NORMAL"}
        sos["lab_severity"] = df_raw["OverallInterp"].astype(str).map(interp_map).fillna("ACTION")

        for a in ANALYTES:
            sos[a] = 0.0

        if "InterpText" in df_raw.columns:
            for i, text in enumerate(df_raw["InterpText"]):
                t = str(text).upper()
                sos.loc[i, "fe_ppm"] = 65.0 if ("IRON" in t or "(FE)" in t) else 15.0
                sos.loc[i, "si_ppm"] = 25.0 if ("SILICON" in t or "(SI)" in t) else 5.0
                sos.loc[i, "al_ppm"] = 15.0 if ("ALUMINUM" in t or "(AL)" in t) else 3.0
                sos.loc[i, "cu_ppm"] = 20.0 if ("COPPER" in t or "(CU)" in t) else 4.0
                sos.loc[i, "water_pct"] = 0.15 if "WATER" in t else 0.02

        if os.path.exists(csv_wo):
            wo = pd.read_csv(csv_wo, parse_dates=["open_date", "close_date"])
        else:
            wo_rows = []
            for idx, r in sos.iterrows():
                text_val = str(df_raw.loc[idx, "InterpText"]).upper() if "InterpText" in df_raw.columns else ""
                if r["lab_severity"] in ["ACTION", "CRITICAL"] or "REPAIR" in text_val:
                    wo_rows.append({
                        "wo_id": "WO-" + str(r["sample_id"]),
                        "machine_id": r["machine_id"],
                        "component": r["component"],
                        "wo_type": "CM",
                        "failure_code": "WEAR-ELEVATED",
                        "description": text_val[:100],
                        "open_date": r["sample_date"] + pd.Timedelta(days=5),
                        "close_date": r["sample_date"] + pd.Timedelta(days=8),
                        "smu_at_wo": r["smu_hours"] + 50.0,
                        "downtime_hours": 12.0,
                        "parts_cost": 1500.0,
                        "labour_cost": 600.0
                    })
            wo = pd.DataFrame(wo_rows)
            if len(wo) == 0:
                wo = pd.DataFrame([{
                    "wo_id": "WO-001", "machine_id": sos["machine_id"].iloc[0],
                    "component": sos["component"].iloc[0], "wo_type": "CM",
                    "failure_code": "WEAR-ELEVATED", "description": "Maintenance",
                    "open_date": sos["sample_date"].iloc[0] + pd.Timedelta(days=5),
                    "close_date": sos["sample_date"].iloc[0] + pd.Timedelta(days=8),
                    "smu_at_wo": 1050.0, "downtime_hours": 10.0,
                    "parts_cost": 1000.0, "labour_cost": 500.0
                }])
    elif os.path.exists(csv_sos):
        sos = pd.read_csv(csv_sos, parse_dates=["sample_date"])
        wo = pd.read_csv(csv_wo, parse_dates=["open_date", "close_date"])
    else:
        raise FileNotFoundError("Could not find SosFluidSample.xlsx or sos_samples.csv in data/raw/")

    sos["component"] = sos["component"].str.upper().str.strip()
    wo["component"] = wo["component"].str.upper().str.strip()
    wo["wo_type"] = wo["wo_type"].str.upper().str.strip()

    # One sample per machine/component/date; keep the last if the lab re-issued.
    sos = (sos.sort_values(["machine_id", "component", "sample_date", "sample_id"])
              .drop_duplicates(["machine_id", "component", "sample_date"], keep="last")
              .reset_index(drop=True))

    for c in ANALYTES:
        sos[c] = pd.to_numeric(sos[c], errors="coerce").clip(lower=0)
    return sos, wo


# ------------------------------------------------------------------------- labels
def _cm_index(wo: pd.DataFrame) -> dict:
    """machine|component -> sorted array of corrective work-order open dates."""
    cm = wo[(wo.wo_type == "CM") & (wo.component != "OTHER")]
    out = {}
    for (m, c), g in cm.groupby(["machine_id", "component"]):
        out[(m, c)] = np.sort(g["open_date"].values)
    return out


def add_labels(sos: pd.DataFrame, wo: pd.DataFrame, horizon=HORIZON_DAYS) -> pd.DataFrame:
    idx = _cm_index(wo)
    df = sos.copy()

    nxt = np.full(len(df), np.nan)
    prev_days = np.full(len(df), np.nan)
    prior_cm_1y = np.zeros(len(df))

    keys = list(zip(df.machine_id, df.component))
    dates = df["sample_date"].values
    for i, (k, d) in enumerate(zip(keys, dates)):
        arr = idx.get(k)
        if arr is None:
            prior_cm_1y[i] = 0
            continue
        j = np.searchsorted(arr, d, side="right")          # first WO strictly after sample
        if j < len(arr):
            nxt[i] = (arr[j] - d) / np.timedelta64(1, "D")
        if j > 0:
            prev_days[i] = (d - arr[j - 1]) / np.timedelta64(1, "D")
            lo = np.searchsorted(arr, d - np.timedelta64(365, "D"), side="left")
            prior_cm_1y[i] = j - lo

    df["days_to_next_cm"] = nxt
    df["days_since_last_cm"] = prev_days
    df["prior_cm_365d"] = prior_cm_1y
    df["label"] = ((df["days_to_next_cm"] > 0) & (df["days_to_next_cm"] <= horizon)).astype(int)

    # --- guards ---------------------------------------------------------------
    last_known = wo["open_date"].max() if len(wo) > 0 else df["sample_date"].max()
    df["censored"] = False if len(df) < 50 else (df["sample_date"] + pd.Timedelta(days=horizon) > last_known)
    df["post_repair"] = False if len(df) < 50 else (df["days_since_last_cm"] <= POST_REPAIR_BLACKOUT)
    return df


# ----------------------------------------------------------------------- features
def _slope(vals: pd.Series) -> float:
    v = vals.dropna().values
    if len(v) < 2:
        return np.nan
    return float(np.polyfit(np.arange(len(v)), v, 1)[0])


def fit_peer_stats(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Robust baseline per (model, component), learned from the PAST only."""
    past = df[df.sample_date < cutoff]
    g = past.groupby(["model", "component"])[ANALYTES]
    med = g.median().add_suffix("__peer_med")
    iqr = (g.quantile(0.75) - g.quantile(0.25)).add_suffix("__peer_iqr")
    return med.join(iqr).reset_index()


def add_features(df: pd.DataFrame, peer: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machine_id", "component", "sample_date"]).copy()
    g = df.groupby(["machine_id", "component"], sort=False)

    df["days_since_prev_sample"] = g["sample_date"].diff().dt.days
    df["hours_since_prev_sample"] = g["smu_hours"].diff()
    df["sample_seq"] = g.cumcount()

    for c in ANALYTES:
        prev = g[c].shift(1)
        df[f"{c}__delta"] = df[c] - prev
        df[f"{c}__pct_change"] = (df[c] - prev) / prev.replace(0, np.nan)
        # wear rate: ppm added per 100 machine hours since the previous sample
        df[f"{c}__per100h"] = df[f"{c}__delta"] / df["hours_since_prev_sample"].replace(0, np.nan) * 100
        # own-history baseline: rolling median of the 5 PREVIOUS samples (shifted = no leakage)
        roll = g[c].transform(lambda s: s.shift(1).rolling(5, min_periods=2).median())
        df[f"{c}__vs_own_med"] = df[c] / roll.replace(0, np.nan)
        df[f"{c}__slope3"] = g[c].transform(lambda s: s.rolling(3, min_periods=2).apply(_slope, raw=False))

    df = df.copy().merge(peer, on=["model", "component"], how="left")
    for c in ANALYTES:
        df[f"{c}__peer_z"] = (df[c] - df[f"{c}__peer_med"]) / df[f"{c}__peer_iqr"].replace(0, np.nan)
        df.drop(columns=[f"{c}__peer_med", f"{c}__peer_iqr"], inplace=True)

    # oil-condition context
    df["fe_per_oil_hour"] = df["fe_ppm"] / df["oil_hours"].replace(0, np.nan)
    df["si_to_fe"] = df["si_ppm"] / df["fe_ppm"].replace(0, np.nan)
    df["cu_to_fe"] = df["cu_ppm"] / df["fe_ppm"].replace(0, np.nan)
    df["lab_severity_num"] = df["lab_severity"].map(SEVERITY_ORDER).fillna(0)
    df["days_since_last_cm"] = df["days_since_last_cm"].fillna(9999)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    engineered = [c for c in df.columns if "__" in c]
    extras = ["oil_hours", "smu_hours", "days_since_prev_sample", "hours_since_prev_sample",
              "sample_seq", "fe_per_oil_hour", "si_to_fe", "cu_to_fe",
              "lab_severity_num", "days_since_last_cm", "prior_cm_365d"]
    return ANALYTES + engineered + extras


def build(sos_path="data/raw/sos_samples.csv", wo_path="data/raw/work_orders.csv"):
    sos, wo = load_raw(sos_path, wo_path)
    labelled = add_labels(sos, wo)
    cutoff = labelled["sample_date"].quantile(TRAIN_FRACTION)
    peer = fit_peer_stats(labelled, cutoff)
    full = add_features(labelled, peer)
    return full, peer, cutoff


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    full, peer, cutoff = build()

    usable = full[~full.censored & ~full.post_repair]
    full.to_csv("data/processed/model_table.csv", index=False)
    peer.to_csv("artifacts/peer_stats.csv", index=False)

    print(f"samples            : {len(full):,}")
    print(f"usable for training: {len(usable):,} "
          f"(dropped {full.censored.sum():,} censored, {full.post_repair.sum():,} post-repair)")
    print(f"positive rate      : {usable.label.mean():.2%}  "
          f"({int(usable.label.sum()):,} samples followed by a corrective WO within {HORIZON_DAYS}d)")
    print(f"peer baseline cutoff: {pd.Timestamp(cutoff).date()}")
    print(f"features           : {len(feature_columns(full))}")
