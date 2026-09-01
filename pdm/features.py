"""
Labelling and feature engineering -- shared by every modelling phase so the
train / score / survival paths can never drift apart.

Labelling (add_labels)
    For each oil sample: did a CORRECTIVE work order land on the same machine +
    compartment within `horizon` days? Plus three leakage guards:
      * PM work orders are never positives (only wo_type == "CM" is indexed).
      * `censored`  - the horizon window runs past the data cut-off.
      * `post_repair` - the sample was taken within POST_REPAIR_BLACKOUT_D days of
        a repair, so the oil is disturbed and the label is unreliable.

Features (add_features)
    Everything is computed *within an oil run* -- the block of samples between two
    oil changes -- so a fresh-oil dilution never looks like a recovery and a
    trend never carries across a drain. Per analyte:
      __delta, __pct_change   change since the previous sample in the run
      __per100h               wear rate: ppm added per 100 machine hours
      __vs_own_med            level vs this machine's own rolling median (shifted)
      __slope3                local 3-point slope
      __ewma                  time-aware EWMA against oil hours (irregular gaps)
      __peer_z                (value - peer median) / peer IQR   [ASTM D7720 style]
      __mass_rate_mg_h        physics: ppm x sump_volume / oil_hours  (a quantity,
                              not a concentration -- top-ups no longer confound it)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

ANALYTES = config.ANALYTES
EWMA_TAU_HOURS = 250.0          # EWMA memory, in oil hours


# --------------------------------------------------------------------- labelling
def _cm_dates(wo: pd.DataFrame) -> dict:
    """(machine_id, component) -> sorted array of corrective WO open dates."""
    cm = wo[(wo["wo_type"] == "CM") & (wo["component"] != "OTHER")]
    return {k: np.sort(g["open_date"].values)
            for k, g in cm.groupby(["machine_id", "component"])}


def add_labels(sos: pd.DataFrame, wo: pd.DataFrame,
               horizon: int = config.PRIMARY_HORIZON) -> pd.DataFrame:
    df = sos.copy()
    idx = _cm_dates(wo)

    n = len(df)
    days_next = np.full(n, np.nan)
    days_prev = np.full(n, np.nan)
    prior_cm = np.zeros(n)

    keys = list(zip(df["machine_id"], df["component"]))
    dates = df["sample_date"].values
    for i, (k, d) in enumerate(zip(keys, dates)):
        arr = idx.get(k)
        if arr is None or len(arr) == 0:
            continue
        j = np.searchsorted(arr, d, side="right")
        if j < len(arr):
            days_next[i] = (arr[j] - d) / np.timedelta64(1, "D")
        if j > 0:
            days_prev[i] = (d - arr[j - 1]) / np.timedelta64(1, "D")
            lo = np.searchsorted(arr, d - np.timedelta64(365, "D"), side="left")
            prior_cm[i] = j - lo

    df["days_to_next_cm"] = days_next
    df["days_since_last_cm"] = days_prev
    df["prior_cm_365d"] = prior_cm
    df["label"] = ((df["days_to_next_cm"] > 0) & (df["days_to_next_cm"] <= horizon)).astype(int)

    data_cutoff = wo["open_date"].max() if len(wo) else df["sample_date"].max()
    small = n < 50
    df["censored"] = False if small else (
        df["sample_date"] + pd.Timedelta(days=horizon) > data_cutoff)
    df["post_repair"] = (df["days_since_last_cm"] <= config.POST_REPAIR_BLACKOUT_D).fillna(False)
    df["usable"] = ~df["censored"] & ~df["post_repair"]
    return df


# ----------------------------------------------------------------- peer baseline
def fit_peer_baseline(sos: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Robust per (model_family, component_family) baseline from the healthy PAST."""
    past = sos[(sos["sample_date"] < cutoff)
               & (sos["lab_severity"].isin(config.HEALTHY_SEVERITIES))]
    if len(past) < 5:                       # too little history -- use all past
        past = sos[sos["sample_date"] < cutoff]
    g = past.groupby(["model_family", "component_family"])[ANALYTES]
    med = g.median().add_suffix("__peer_med")
    iqr = (g.quantile(0.75) - g.quantile(0.25)).add_suffix("__peer_iqr")
    return med.join(iqr).reset_index()


# --------------------------------------------------------------------- features
def _run_trend_features(vals: np.ndarray, oil_hours: np.ndarray,
                        tau: float) -> dict[str, np.ndarray]:
    """Within-oil-run trend features for one machine/compartment run.

    `vals` is (n_samples, n_analytes) in time order. Everything here is causal --
    it only ever looks at earlier samples in the same run -- so a fresh-oil
    dilution never reads as a recovery and no trend carries across a drain.
    """
    n, k = vals.shape
    prev = np.full((n, k), np.nan)
    prev[1:] = vals[:-1]
    delta = vals - prev

    # rolling median of up to the 5 PREVIOUS samples (shifted -> no leakage)
    own_med = np.full((n, k), np.nan)
    for i in range(1, n):
        window = vals[max(0, i - 5):i]
        with np.errstate(invalid="ignore"):
            m = np.nanmedian(window, axis=0) if len(window) else np.full(k, np.nan)
        own_med[i] = m

    # 3-sample local slope: mean of the last two step changes
    slope3 = np.full((n, k), np.nan)
    if n >= 2:
        step = delta.copy()
        slope3[1] = step[1]
        for i in range(2, n):
            slope3[i] = np.nanmean(step[i - 1:i + 1], axis=0)

    # time-aware EWMA: decay set by the oil-hour gap between samples
    ewma = np.full((n, k), np.nan)
    running = None
    prev_h = np.nan
    for i in range(n):
        v = vals[i]
        h = oil_hours[i]
        if running is None:
            running = np.where(np.isnan(v), np.nan, v)
        else:
            dh = h - prev_h if (np.isfinite(h) and np.isfinite(prev_h) and h >= prev_h) else tau
            alpha = 1.0 - np.exp(-max(float(dh), 1e-6) / tau)
            step_val = alpha * v + (1 - alpha) * running
            running = np.where(np.isnan(v), running, step_val)
        ewma[i] = running
        if np.isfinite(h):
            prev_h = h
    return {"prev": prev, "delta": delta, "own_med": own_med,
            "slope3": slope3, "ewma": ewma}


def add_features(sos: pd.DataFrame, peer: pd.DataFrame) -> pd.DataFrame:
    df = sos.sort_values(["machine_id", "component", "sample_date"]).copy()

    # oil run id: increments on every recorded fluid change
    fc = df.groupby(["machine_id", "component"])["fluid_changed"]
    df["oil_run"] = fc.transform(lambda s: s.shift(1, fill_value=False).cumsum())

    grp = ["machine_id", "component", "oil_run"]
    g = df.groupby(grp, sort=False)

    new: dict[str, object] = {
        "sample_seq": df.groupby(["machine_id", "component"]).cumcount(),
        "run_seq": g.cumcount(),
        "days_since_prev_sample": g["sample_date"].diff().dt.days,
        "hours_since_prev_sample": g["smu_hours"].diff(),
    }
    hours_prev = new["hours_since_prev_sample"].replace(0, np.nan).to_numpy()

    peer_lut = peer.set_index(["model_family", "component_family"]) if len(peer) else None
    oh = df["oil_hours"].fillna(df["smu_hours"]).to_numpy(dtype=float)
    amat = df[ANALYTES].to_numpy(dtype=float)

    # one causal pass per oil run, all analytes together (vectorised over analytes)
    K = len(ANALYTES)
    prev_m = np.full((len(df), K), np.nan)
    delta_m = np.full((len(df), K), np.nan)
    ownmed_m = np.full((len(df), K), np.nan)
    slope_m = np.full((len(df), K), np.nan)
    ewma_m = np.full((len(df), K), np.nan)
    for _, pos in df.groupby(grp, sort=False).indices.items():
        sl = np.sort(pos)
        r = _run_trend_features(amat[sl], oh[sl], EWMA_TAU_HOURS)
        prev_m[sl], delta_m[sl] = r["prev"], r["delta"]
        ownmed_m[sl], slope_m[sl], ewma_m[sl] = r["own_med"], r["slope3"], r["ewma"]

    keys = list(zip(df["model_family"], df["component_family"]))
    for j, c in enumerate(ANALYTES):
        prev = prev_m[:, j]
        delta = delta_m[:, j]
        with np.errstate(invalid="ignore", divide="ignore"):
            new[f"{c}__delta"] = delta
            new[f"{c}__pct_change"] = delta / np.where(prev == 0, np.nan, prev)
            new[f"{c}__per100h"] = delta / hours_prev * 100
            new[f"{c}__vs_own_med"] = amat[:, j] / np.where(ownmed_m[:, j] == 0, np.nan, ownmed_m[:, j])
        new[f"{c}__slope3"] = slope_m[:, j]
        new[f"{c}__ewma"] = ewma_m[:, j]
        new[f"{c}__ewma_dev"] = amat[:, j] - ewma_m[:, j]

        if peer_lut is not None and f"{c}__peer_med" in peer_lut.columns:
            med = peer_lut[f"{c}__peer_med"].reindex(keys).to_numpy()
            iqr = peer_lut[f"{c}__peer_iqr"].reindex(keys).to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                new[f"{c}__peer_z"] = (amat[:, j] - med) / np.where(iqr == 0, np.nan, iqr)
        else:
            new[f"{c}__peer_z"] = np.nan

    # physics: concentration -> wear-metal mass generated per operating hour
    oil_h = df["oil_hours"].where(df["oil_hours"] > 0).fillna(
        df["smu_hours"].clip(lower=1)).clip(lower=1)
    for m in config.WEAR_METALS:
        new[f"{m}__mass_rate_mg_h"] = df[m] * df["sump_volume_l"] / oil_h

    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # contextual ratios
    df["fe_per_oil_hour"] = df["fe_ppm"] / oil_h
    df["si_to_fe"] = df["si_ppm"] / df["fe_ppm"].replace(0, np.nan)
    df["cu_to_fe"] = df["cu_ppm"] / df["fe_ppm"].replace(0, np.nan)
    df["oil_age_frac"] = oil_h / oil_h.groupby([df["component_family"]]).transform("median")
    df["lab_severity_num"] = df["lab_severity"].map(config.SEVERITY_ORDER).fillna(0)
    df["days_since_last_cm"] = df["days_since_last_cm"].fillna(9999)
    df["prior_cm_365d"] = df.get("prior_cm_365d", pd.Series(0, index=df.index)).fillna(0)

    df = mine_interp_text(df)
    return df.reset_index(drop=True)


# --------------------------------------------------------------- InterpText NLP
_TEXT_FLAGS = {
    "flag__iron":        r"\bIRON\b|\(FE\)",
    "flag__dirt":        r"\bSILICON\b|\bDIRT\b|\(SI\)|INDUCTION|BREATHER",
    "flag__coolant":     r"\bCOOLANT\b|\bGLYCOL\b|\(GLY\)",
    "flag__water":       r"\bWATER\b",
    "flag__fuel":        r"\bFUEL\b DILUTION|FUEL DILUTION",
    "flag__bearing":     r"\bBEARING\b|\bLEAD\b|\(PB\)",
    "flag__repair_rec":  r"\bREPAIR\b|\bINSPECT\b|\bSTOP\b|CHANGE OIL",
    "flag__resample_rec": r"RESAMPLE|SAMPLE IN \d+|MONITOR",
}


def mine_interp_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = df.get("interp_text", pd.Series("", index=df.index)).astype(str).str.upper()
    for flag, pat in _TEXT_FLAGS.items():
        df[flag] = text.str.contains(pat, regex=True, na=False).astype(int)
    df["flag__text_present"] = (text.str.len() > 3).astype(int)
    return df


# ------------------------------------------------------------- feature selection
def feature_columns(df: pd.DataFrame) -> list[str]:
    engineered = [c for c in df.columns if "__" in c and not c.endswith(("__peer_med", "__peer_iqr"))]
    flags = [c for c in df.columns if c.startswith("flag__")]
    extras = ["oil_hours", "smu_hours", "makeup_fluid_l", "sample_seq", "run_seq",
              "days_since_prev_sample", "hours_since_prev_sample",
              "fe_per_oil_hour", "si_to_fe", "cu_to_fe", "oil_age_frac",
              "lab_severity_num", "days_since_last_cm", "prior_cm_365d"]
    cols = ANALYTES + engineered + flags + [c for c in extras if c in df.columns]
    # keep only columns that actually carry signal in this extract
    return [c for c in dict.fromkeys(cols) if c in df.columns and df[c].notna().any()]


def monotone_constraints(features: list[str]) -> list[int]:
    """+1 where more of the quantity must not decrease predicted risk."""
    worse = set(config.HIGHER_IS_WORSE)
    out = []
    for f in features:
        base = f.split("__")[0]
        rising = f.endswith(("__delta", "__per100h", "__ewma_dev", "__peer_z",
                             "__vs_own_med", "__slope3", "__mass_rate_mg_h"))
        if f in ("lab_severity_num", "prior_cm_365d") or (base in worse and (rising or f == base)):
            out.append(1)
        else:
            out.append(0)
    return out
