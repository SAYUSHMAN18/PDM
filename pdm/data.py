"""
Load the raw extracts and canonicalise them into the internal schema every other
module expects. This is the ONLY module that knows what the source columns are
called.

Three data modes, decided automatically and printed loudly:

  SYNTHETIC         data/raw/sos_samples.csv exists and the .synthetic marker is
                    present -- the generator produced it.
  REAL_FULL         sos_samples.csv + work_orders.csv present, no marker -- real
                    extracts dropped in with the contract column names.
  REAL_SAMPLE_ONLY  only the 8-row SosFluidSample.xlsx is present. It has no
                    numeric chemistry and no work orders, so phases 3-5 cannot
                    train. We do NOT fabricate values to paper over that.

Canonical S.O.S columns
    sample_id machine_id serial model model_family site
    component component_family position sump_volume_l
    sample_date processed_date smu_hours oil_hours fluid_changed makeup_fluid_l
    lab_severity lab_code interp_text wo_ref
    + the 17 analytes from config.ANALYTES

Canonical work-order columns
    wo_id machine_id component component_family wo_type failure_code description
    open_date close_date smu_at_wo downtime_h parts_cost labour_cost
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

# raw S.O.S column -> canonical analyte
_ANALYTE_SOURCE = {
    "fe_ppm": ["Fe", "IRON", "Iron", "fe_ppm"],
    "cu_ppm": ["Cu", "COPPER", "Copper", "cu_ppm"],
    "cr_ppm": ["Cr", "CHROMIUM", "Chromium", "cr_ppm"],
    "pb_ppm": ["Pb", "LEAD", "Lead", "pb_ppm"],
    "al_ppm": ["Al", "ALUMINUM", "Aluminium", "al_ppm"],
    "si_ppm": ["Si", "SILICON", "Silicon", "si_ppm"],
    "na_ppm": ["Na", "SODIUM", "Sodium", "na_ppm"],
    "k_ppm": ["K", "POTASSIUM", "Potassium", "k_ppm"],
    "water_pct": ["Water", "WATER", "water_pct", "WaterPercent"],
    "fuel_pct": ["Fuel", "FUEL", "fuel_pct", "FuelDilution"],
    "glycol_pct": ["Glycol", "GLYCOL", "glycol_pct"],
    "soot_pct": ["Soot", "SOOT", "soot_pct"],
    "visc100": ["Visc100", "Viscosity100", "VISC100", "visc100", "Visc40", "visc40"],
    "oxidation": ["Oxidation", "OXIDATION", "oxidation"],
    "nitration": ["Nitration", "NITRATION", "nitration"],
    "tbn": ["TBN", "tbn", "Tbn"],
    "pq_index": ["PQ", "PQIndex", "pq_index", "PQ_Index"],
}


@dataclass
class Extracts:
    sos: pd.DataFrame
    wo: pd.DataFrame
    assets: pd.DataFrame
    mode: str

    @property
    def has_work_orders(self) -> bool:
        return len(self.wo) > 0 and (self.wo["wo_type"] == "CM").any()

    @property
    def has_chemistry(self) -> bool:
        return bool(self.sos[config.WEAR_METALS].notna().any().any())


# ---------------------------------------------------------------------- helpers
def _first_col(df: pd.DataFrame, names) -> pd.Series | None:
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def _severity_from_code(code) -> str:
    return config.INTERP_CODE_MAP.get(str(code).upper().strip(), "NORMAL")


def _clean_component(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip().str.replace(r"\s+", "_", regex=True)


# ------------------------------------------------------------------ asset master
def _load_assets() -> pd.DataFrame:
    if config.ASSET_CSV.exists():
        am = pd.read_csv(config.ASSET_CSV)
        rename = {"EquipNum": "machine_id", "TMSAssetID": "tms_asset_id",
                  "SerialNum": "serial", "EqpModel": "model",
                  "ModelFamily": "model_family", "SiteId": "site_id",
                  "SiteName": "site", "Status": "status"}
        am = am.rename(columns={k: v for k, v in rename.items() if k in am.columns})
        am["machine_id"] = am["machine_id"].astype(str)
        return am
    return pd.DataFrame(columns=["machine_id", "serial", "model", "model_family", "site"])


def _model_family(model: pd.Series) -> pd.Series:
    m = model.astype(str).str.upper()
    fam = np.where(m.str.contains("988"), "CAT_988",
          np.where(m.str.contains("773"), "CAT_773",
          np.where(m.str.contains("777"), "CAT_777",
          np.where(m.str.contains("336"), "CAT_336",
          np.where(m.str.contains("349"), "CAT_349", "OTHER")))))
    return pd.Series(fam, index=model.index)


# -------------------------------------------------------------------- S.O.S load
def _canonicalise_sos(raw: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)

    sid = _first_col(raw, ["SampleNum", "SampleNumber", "sample_id", "Id"])
    df["sample_id"] = (sid.astype(str) if sid is not None
                       else [f"S{i:08d}" for i in range(len(raw))])

    mid = _first_col(raw, ["EquipNum", "EquipmentNumber", "machine_id", "AssetName"])
    ser = _first_col(raw, ["SerialNum", "SerialNumber", "serial"])
    df["machine_id"] = mid.astype(str) if mid is not None else ser.astype(str)
    df["serial"] = ser.astype(str) if ser is not None else ""

    model = _first_col(raw, ["EqpModel", "Model", "model"])
    df["model"] = model.astype(str) if model is not None else "UNKNOWN"
    mf = _first_col(raw, ["ModelFamily", "model_family"])
    df["model_family"] = mf.astype(str) if mf is not None else _model_family(df["model"])

    site = _first_col(raw, ["SiteName", "JobsiteDesc", "EqpJobsite", "site"])
    df["site"] = site.astype(str) if site is not None else "UNKNOWN"

    comp = _first_col(raw, ["Compartment", "Component", "component"])
    df["component"] = _clean_component(comp)
    df["component_family"] = df["component"].map(config.compartment_family)
    df["position"] = df["component"].map(config.compartment_position)
    df["sump_volume_l"] = df["component"].map(config.compartment_sump_l)

    sd = _first_col(raw, ["DateSampled", "SampleDate", "sample_date"])
    cd = _first_col(raw, ["CreatedDate", "created_date"])
    df["sample_date"] = pd.to_datetime(sd, errors="coerce")
    if cd is not None:
        df["sample_date"] = df["sample_date"].fillna(pd.to_datetime(cd, errors="coerce"))
    pd_ = _first_col(raw, ["DateProcessed", "ProcessedDate", "processed_date"])
    df["processed_date"] = pd.to_datetime(pd_, errors="coerce") if pd_ is not None else pd.NaT

    smu = _first_col(raw, ["CMeter", "SMU", "smu_hours", "CompMeter"])
    oil = _first_col(raw, ["CMeterFluid", "OilHours", "oil_hours"])
    df["smu_hours"] = pd.to_numeric(smu, errors="coerce") if smu is not None else np.nan
    df["oil_hours"] = pd.to_numeric(oil, errors="coerce") if oil is not None else np.nan

    fc = _first_col(raw, ["FluidChanged", "fluid_changed", "OilChanged"])
    df["fluid_changed"] = (fc.astype(str).str.upper().str.strip().isin(["Y", "YES", "TRUE", "1"])
                           if fc is not None else False)
    mu = _first_col(raw, ["MakeUpFluid", "makeup_fluid_l"])
    df["makeup_fluid_l"] = pd.to_numeric(mu, errors="coerce").fillna(0.0) if mu is not None else 0.0

    code = _first_col(raw, ["OverallInterp", "InterpCode", "lab_code"])
    df["lab_code"] = code.astype(str).str.upper().str.strip() if code is not None else "U"
    sev = _first_col(raw, ["lab_severity", "Severity"])
    df["lab_severity"] = (sev.astype(str).str.upper() if sev is not None
                          else df["lab_code"].map(_severity_from_code))

    it = _first_col(raw, ["InterpText", "Interpretation", "interp_text"])
    df["interp_text"] = it.astype(str) if it is not None else ""
    wr = _first_col(raw, ["WorkOrderId", "WorkOrderID", "wo_ref"])
    df["wo_ref"] = wr.astype(str) if wr is not None else ""

    for canon, sources in _ANALYTE_SOURCE.items():
        col = _first_col(raw, sources)
        df[canon] = pd.to_numeric(col, errors="coerce") if col is not None else np.nan

    # attach model / family from asset master where the sample lacked it
    if len(assets) and "model_family" in assets.columns:
        lut = assets.drop_duplicates("machine_id").set_index("machine_id")
        for c in ("model", "model_family", "site"):
            if c in lut.columns:
                fill = df["machine_id"].map(lut[c])
                df[c] = df[c].where(~df[c].isin(["UNKNOWN", "nan", ""]), fill)

    df["model_family"] = df["model_family"].fillna("OTHER").replace({"nan": "OTHER", "": "OTHER"})

    # one sample per machine / compartment / date; keep the last re-issue
    df = (df.sort_values(["machine_id", "component", "sample_date", "sample_id"])
            .drop_duplicates(["machine_id", "component", "sample_date"], keep="last")
            .reset_index(drop=True))
    for a in config.ANALYTES:
        df[a] = pd.to_numeric(df[a], errors="coerce").clip(lower=0)
    return df


# -------------------------------------------------------------- work-order load
def _canonicalise_wo(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=[
            "wo_id", "machine_id", "component", "component_family", "wo_type",
            "failure_code", "description", "open_date", "close_date",
            "smu_at_wo", "downtime_h", "parts_cost", "labour_cost"])

    df = pd.DataFrame(index=raw.index)
    df["wo_id"] = _first_col(raw, ["WorkOrderId", "WOId", "wo_id"]).astype(str)
    df["machine_id"] = _first_col(raw, ["EquipNum", "machine_id", "AssetId"]).astype(str)
    comp = _first_col(raw, ["Compartment", "Component", "component"])
    df["component"] = _clean_component(comp) if comp is not None else "OTHER"
    df["component_family"] = df["component"].map(config.compartment_family)

    wt = _first_col(raw, ["WOType", "Type", "wo_type", "MaintenanceType"])
    wt = wt.astype(str).str.upper().str.strip() if wt is not None else pd.Series("CM", index=raw.index)
    df["wo_type"] = np.where(wt.str.startswith("P") | wt.str.contains("PREV") | wt.str.contains("SCHEDUL"),
                             "PM",
                    np.where(wt.str.startswith("C") | wt.str.contains("CORR") | wt.str.contains("BREAK") |
                             wt.str.contains("REPAIR") | wt.str.contains("FAIL"), "CM", "CM"))

    fc = _first_col(raw, ["FailureCode", "failure_code", "ProblemCode"])
    df["failure_code"] = fc.astype(str) if fc is not None else ""
    de = _first_col(raw, ["Description", "description", "WODescription"])
    df["description"] = de.astype(str) if de is not None else ""

    df["open_date"] = pd.to_datetime(_first_col(raw, ["OpenDate", "open_date", "CreatedDate"]), errors="coerce")
    df["close_date"] = pd.to_datetime(_first_col(raw, ["CloseDate", "close_date", "CompletedDate"]), errors="coerce")

    for canon, names in {"smu_at_wo": ["SMUAtWO", "smu_at_wo", "SMU"],
                         "downtime_h": ["DowntimeHours", "downtime_h", "Downtime"],
                         "parts_cost": ["PartsCost", "parts_cost"],
                         "labour_cost": ["LabourCost", "labour_cost", "LaborCost"]}.items():
        col = _first_col(raw, names)
        df[canon] = pd.to_numeric(col, errors="coerce") if col is not None else np.nan

    return df.dropna(subset=["open_date"]).reset_index(drop=True)


# ---------------------------------------------------------------------- public
def load_extracts(verbose: bool = True) -> Extracts:
    assets = _load_assets()

    if config.SOS_CSV.exists():
        raw_sos = pd.read_csv(config.SOS_CSV)
        raw_wo = pd.read_csv(config.WO_CSV) if config.WO_CSV.exists() else None
        mode = "SYNTHETIC" if config.SYNTHETIC_MARKER.exists() else "REAL_FULL"
    elif config.SOS_XLSX.exists():
        raw_sos = pd.read_excel(config.SOS_XLSX)
        raw_wo = None
        mode = "REAL_SAMPLE_ONLY"
    else:
        raise FileNotFoundError(
            f"No S.O.S data found. Expected one of:\n"
            f"  {config.SOS_CSV}  (real or synthetic)\n"
            f"  {config.SOS_XLSX}  (the small real sample)\n"
            f"Run  python run_pipeline.py --synth  to generate a demo dataset.")

    sos = _canonicalise_sos(raw_sos, assets)
    wo = _canonicalise_wo(raw_wo)

    if len(assets) == 0:
        assets = (sos[["machine_id", "serial", "model", "model_family", "site"]]
                  .drop_duplicates("machine_id").reset_index(drop=True))

    ext = Extracts(sos=sos, wo=wo, assets=assets, mode=mode)
    if verbose:
        _print_summary(ext)
    return ext


def _print_summary(ext: Extracts) -> None:
    s = ext.sos
    print(f"  data mode        : {ext.mode}")
    print(f"  S.O.S samples    : {len(s):,}  "
          f"({s['sample_date'].min():%Y-%m-%d} -> {s['sample_date'].max():%Y-%m-%d})")
    print(f"  machines         : {s['machine_id'].nunique()}   "
          f"compartments: {s['component'].nunique()}   "
          f"families: {sorted(s['component_family'].unique())}")
    print(f"  work orders      : {len(ext.wo):,}  "
          f"(CM {int((ext.wo['wo_type'] == 'CM').sum()):,} / PM {int((ext.wo['wo_type'] == 'PM').sum()):,})")
    if not ext.has_chemistry:
        print("  WARNING: no numeric chemistry in this extract -- phases 3-5 will be "
              "skipped / degraded. Phase 2 runs on lab severity + InterpText only.")
    if not ext.has_work_orders:
        print("  WARNING: no corrective work orders -- phases 3-5 need labels and will be skipped.")


if __name__ == "__main__":
    load_extracts()
