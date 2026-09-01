"""
Realistic synthetic S.O.S + work-order + asset data, written in the *enterprise
extract schema* so the rest of the pipeline treats it exactly like the real
files that are on the way.

What it models
--------------
* A mixed CAT fleet (988 wheel loaders, 773 haul trucks, 336 excavators), each
  with several oil-sampled compartments.
* Wear metals that accumulate with oil age and reset at every oil change.
* 0-3 fault episodes per machine-compartment, each ramping toward a corrective
  work order, with a few days of reporting lag.
* Scheduled oil-change PM work orders, and some unrelated corrective jobs so the
  CMMS is not unrealistically tidy.
* Real-world mess: ~15% of samples missing oil hours, occasional missing SMU.

Output (data/raw/):
    sos_samples.csv      work_orders.csv      asset_master.csv
    .synthetic           (marker so data.py can label the run SYNTHETIC)

When the real extracts arrive, delete these three CSVs and drop the real ones in
with the same names. Column contract is documented in DATA_REQUEST.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

RNG_SEED = 20260827
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2026-06-30")

# ------------------------------------------------------------------ fleet layout
# (model, model_family, [compartment codes], site)
_SITES = [
    ("024901", "Mansfield OH"),
    ("024902", "Sweetwater TX"),
    ("024903", "Gillette WY"),
]
_WHEEL_LOADER_COMPS = ["ENGINE", "TRANSMISSION", "HYDRAULIC", "DIFF_FR", "DIFF_RR",
                       "FD_FR_LT", "FD_FR_RT", "FD_RR_LT", "FD_RR_RT"]
_HAUL_TRUCK_COMPS = ["ENGINE", "TRANSMISSION", "HYDRAULIC", "DIFF_RR", "FD_RR_LT", "FD_RR_RT"]
_EXCAVATOR_COMPS = ["ENGINE", "HYDRAULIC", "SWING", "FD_LT", "FD_RT"]

_MODELS = [
    ("988F_CAT", "CAT_988", _WHEEL_LOADER_COMPS, 14),
    ("988G_CAT", "CAT_988", _WHEEL_LOADER_COMPS, 10),
    ("773G_CAT", "CAT_773", _HAUL_TRUCK_COMPS, 12),
    ("336F_CAT", "CAT_336", _EXCAVATOR_COMPS, 10),
]

# ---------------------------------------------------- healthy baseline chemistry
# per component family; fresh-ish oil, healthy machine.
_BASE = {
    "ENGINE":       dict(fe=17, cu=7, cr=2.0, pb=4, al=5, si=8, na=4, k=3,
                         water=0.02, fuel=0.6, glycol=0.0, soot=0.4,
                         visc=13.8, oxid=12, nitr=8, tbn=9.0, pq=8, oil_life_h=500),
    "TRANSMISSION": dict(fe=36, cu=13, cr=3.0, pb=6, al=5, si=7, na=3, k=2,
                         water=0.03, fuel=0.0, glycol=0.0, soot=0.1,
                         visc=15.5, oxid=9, nitr=5, tbn=6.0, pq=14, oil_life_h=1000),
    "HYDRAULIC":    dict(fe=11, cu=4, cr=1.0, pb=2, al=3, si=6, na=3, k=2,
                         water=0.03, fuel=0.0, glycol=0.0, soot=0.05,
                         visc=9.5, oxid=8, nitr=4, tbn=4.0, pq=6, oil_life_h=2000),
    "DIFFERENTIAL": dict(fe=70, cu=10, cr=4.5, pb=8, al=6, si=9, na=3, k=2,
                         water=0.04, fuel=0.0, glycol=0.0, soot=0.05,
                         visc=18.5, oxid=7, nitr=4, tbn=5.0, pq=26, oil_life_h=2000),
    "FINAL_DRIVE":  dict(fe=80, cu=9, cr=5.0, pb=8, al=6, si=9, na=3, k=2,
                         water=0.05, fuel=0.0, glycol=0.0, soot=0.05,
                         visc=18.0, oxid=7, nitr=4, tbn=5.0, pq=30, oil_life_h=2000),
    "WHEEL_END":    dict(fe=55, cu=8, cr=4.0, pb=7, al=5, si=8, na=3, k=2,
                         water=0.05, fuel=0.0, glycol=0.0, soot=0.05,
                         visc=18.0, oxid=6, nitr=3, tbn=5.0, pq=22, oil_life_h=2500),
    "SWING_DRIVE":  dict(fe=45, cu=7, cr=3.5, pb=6, al=5, si=8, na=3, k=2,
                         water=0.04, fuel=0.0, glycol=0.0, soot=0.05,
                         visc=15.0, oxid=6, nitr=3, tbn=5.0, pq=18, oil_life_h=2000),
}

# fault archetypes: (name, {analyte: peak multiplier / additive}, failure code, description)
_FAULTS = {
    "ENGINE": [
        ("BEARING_WEAR", dict(pb=6.0, cu=4.0, fe=2.2, pq=3.0), "ENG-BRG", "Main / con-rod bearing wear"),
        ("DIRT_INGRESS", dict(si=7.0, al=3.5, fe=2.0), "ENG-AIR", "Air induction leak / dirt ingress"),
        ("COOLANT_LEAK", dict(glycol=1.2, na=6.0, k=6.0, water=9.0), "ENG-COOL", "Coolant entering crankcase"),
        ("FUEL_DILUTION", dict(fuel=7.0, visc=-0.30, soot=2.2), "ENG-INJ", "Injector leak / fuel dilution"),
    ],
    "TRANSMISSION": [
        ("CLUTCH_WEAR", dict(fe=3.0, cr=3.0, pq=4.5), "TRN-CLU", "Clutch pack / gear wear"),
        ("BEARING_WEAR", dict(cu=4.5, pb=5.0, fe=2.2), "TRN-BRG", "Transmission bearing wear"),
    ],
    "HYDRAULIC": [
        ("PUMP_WEAR", dict(fe=3.4, cu=3.6, cr=2.6, pq=4.2), "HYD-PMP", "Hydraulic pump wear"),
        ("SEAL_DIRT", dict(si=8.0, fe=2.4, al=3.0), "HYD-SEAL", "Cylinder seal / dirt ingress"),
        ("WATER_INGRESS", dict(water=13.0, fe=2.0), "HYD-WTR", "Water contamination"),
    ],
    "DIFFERENTIAL": [
        ("GEAR_SCUFF", dict(fe=3.0, cr=3.4, pq=4.6), "DIF-GER", "Crown & pinion scuffing"),
    ],
    "FINAL_DRIVE": [
        ("GEAR_WEAR", dict(fe=3.0, cr=3.5, pq=5.0), "FDR-GER", "Final drive gear wear"),
        ("SEAL_LEAK", dict(si=6.0, water=8.0, al=2.5, fe=1.8), "FDR-SEAL", "Duo-cone seal leak / dirt ingress"),
    ],
    "WHEEL_END": [
        ("PLANETARY_SPALL", dict(fe=3.2, cu=3.0, pq=5.2), "WHE-PLN", "Planetary gear / bearing spalling"),
    ],
    "SWING_DRIVE": [
        ("GEAR_WEAR", dict(fe=3.0, cr=3.0, pq=4.5), "SWG-GER", "Swing drive gear wear"),
    ],
}

_INTERP_TEMPLATES = {
    "fe": "IRON (Fe) IS ELEVATED. CHECK SCREENS / FILTERS / PLUGS FOR ABNORMAL WEAR DEBRIS. RESAMPLE TO MONITOR.",
    "si": "SILICON (Si){al_note} INDICATES DIRT ENTRY. CHECK AIR INDUCTION / SEALS / BREATHERS. CHANGE OIL AND RESAMPLE.",
    "cu": "COPPER (Cu) IS ELEVATED. POSSIBLE BUSHING / THRUST WASHER WEAR. MONITOR CLOSELY.",
    "pb": "LEAD (Pb) AND COPPER (Cu) SUGGEST BEARING WEAR. INSPECT AND RESAMPLE AT 125 HOURS.",
    "water": "WATER CONTAMINATION DETECTED. LOCATE AND CORRECT INGRESS SOURCE. CHANGE OIL.",
    "glycol": "GLYCOL PRESENT - COOLANT LEAK INTO OIL. STOP AND REPAIR BEFORE FURTHER OPERATION.",
    "fuel": "FUEL DILUTION IS HIGH AND VISCOSITY IS LOW. CHECK INJECTORS / PUMP TIMING.",
    "ok": "ALL ANALYSIS READINGS ARE WITHIN ACCEPTABLE LIMITS FOR THIS COMPONENT AND OIL HOURS.",
}


def _severity(row: dict, fam: str) -> str:
    b = _BASE[fam]
    score = 0
    for k in ("fe", "cu", "cr", "pb", "al", "si", "pq"):
        r = row[k] / max(b[k], 1e-6)
        score += 2 if r > 3 else 1 if r > 1.8 else 0
    if row["water"] > 0.5:
        score += 3
    if row["glycol"] > 0.1:
        score += 5
    if row["fuel"] > 5:
        score += 2
    if abs(row["visc"] - b["visc"]) / b["visc"] > 0.15:
        score += 2
    return ("CRITICAL" if score >= 9 else "ACTION" if score >= 5
            else "MONITOR" if score >= 2 else "NORMAL")


_SEV_TO_CODE = {"NORMAL": "A", "MONITOR": "B", "ACTION": "AR", "CRITICAL": "CR"}


def _interp_text(row: dict, fam: str, severity: str) -> str:
    if severity == "NORMAL":
        return _INTERP_TEMPLATES["ok"]
    b = _BASE[fam]
    ratios = {k: row[k] / max(b[k], 1e-6) for k in ("fe", "cu", "pb", "si")}
    if row["glycol"] > 0.1:
        return _INTERP_TEMPLATES["glycol"]
    if row["water"] > 0.5:
        return _INTERP_TEMPLATES["water"]
    if row["fuel"] > 5:
        return _INTERP_TEMPLATES["fuel"]
    if ratios["si"] > 1.8:
        al_note = " AND ALUMINIUM (Al)" if row["al"] / max(b["al"], 1e-6) > 1.8 else ""
        return _INTERP_TEMPLATES["si"].format(al_note=al_note)
    if ratios["pb"] > 1.8:
        return _INTERP_TEMPLATES["pb"]
    if ratios["cu"] > 1.8:
        return _INTERP_TEMPLATES["cu"]
    return _INTERP_TEMPLATES["fe"]


def generate(seed: int = RNG_SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    assets, samples, work_orders = [], [], []
    sample_seq = wo_seq = 0

    # ---- build the fleet ----
    fleet = []
    equip_seq = 50
    for model, family, comps, count in _MODELS:
        for _ in range(count):
            equip_seq += 1
            equip_num = f"120-{equip_seq:06d}" if family == "CAT_988" else f"100-{equip_seq:06d}"
            site_id, site_name = _SITES[rng.integers(len(_SITES))]
            serial = "".join(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ"), 3)) + f"{rng.integers(100, 999)}{rng.integers(100, 999)}"
            commission = START - pd.Timedelta(days=float(rng.uniform(200, 3200)))
            assets.append(dict(
                EquipNum=equip_num, TMSAssetID=equip_num, SerialNum=serial,
                EqpModel=model, ModelFamily=family, SiteId=site_id, SiteName=site_name,
                Status="ACTIVE", CommissionDate=commission.normalize().date(),
            ))
            fleet.append((equip_num, serial, model, family, comps, site_id, site_name))

    # ---- per machine / compartment time series ----
    for equip_num, serial, model, family, comps, site_id, site_name in fleet:
        hours_per_day = float(np.clip(rng.normal(9.0, 2.0), 4.0, 14.0))
        # SMU is a machine-level clock: one reading shared by every compartment,
        # a function of calendar time -- so it is monotonic across the machine.
        smu0 = float(rng.uniform(400, 11000))

        def _smu_at(ts: pd.Timestamp) -> float:
            return smu0 + max((ts - START).days, 0) * hours_per_day

        for comp in comps:
            fam = config.compartment_family(comp)
            base = _BASE[fam]
            oil_life_h = base["oil_life_h"]
            interval_days = float(np.clip(oil_life_h / hours_per_day / 2.0, 20, 95))
            oil_hours = float(rng.uniform(0, oil_life_h))
            t = START + pd.Timedelta(days=float(rng.uniform(0, 25)))

            episodes = []
            for _ in range(int(rng.integers(0, 4))):
                onset = START + pd.Timedelta(days=float(rng.uniform(40, (END - START).days - 50)))
                dur = float(rng.uniform(50, 170))
                name, mult, code, desc = _FAULTS[fam][rng.integers(len(_FAULTS[fam]))]
                episodes.append(dict(onset=onset, repair=onset + pd.Timedelta(days=dur),
                                     mult=mult, code=code, desc=desc, name=name, done=False))
            episodes.sort(key=lambda e: e["onset"])

            while t < END:
                step = float(np.clip(rng.normal(interval_days, interval_days * 0.18), 12, 100))
                t += pd.Timedelta(days=step)
                if t >= END:
                    break
                smu = _smu_at(t)
                oil_hours += step * hours_per_day
                fluid_changed = "N"
                makeup_l = round(float(rng.uniform(0, 4)), 1)

                if oil_hours >= oil_life_h:
                    oil_hours = float(rng.uniform(0, 50))
                    fluid_changed = "Y"
                    makeup_l = 0.0
                    wo_seq += 1
                    work_orders.append(dict(
                        WorkOrderId=f"WO{wo_seq:07d}", EquipNum=equip_num, Compartment=comp,
                        WOType="PM", FailureCode="PM-OIL",
                        Description=f"Scheduled {fam.lower()} oil and filter change",
                        OpenDate=t.normalize().date(),
                        CloseDate=(t + pd.Timedelta(days=1)).normalize().date(),
                        SMUAtWO=round(_smu_at(t), 1), DowntimeHours=round(float(rng.uniform(2, 6)), 1),
                        PartsCost=round(float(rng.uniform(150, 900)), 2),
                        LabourCost=round(float(rng.uniform(100, 400)), 2),
                    ))

                oil_frac = oil_hours / oil_life_h
                row = {}
                for k in ("fe", "cu", "cr", "pb", "al", "si", "na", "k", "pq"):
                    row[k] = base[k] * (0.55 + 0.9 * oil_frac) * float(rng.lognormal(0, 0.16))
                row["visc"] = base["visc"] * float(rng.normal(1.0, 0.025))
                row["oxid"] = base["oxid"] * (0.7 + 0.7 * oil_frac) * float(rng.normal(1.0, 0.08))
                row["nitr"] = base["nitr"] * (0.7 + 0.7 * oil_frac) * float(rng.normal(1.0, 0.08))
                row["soot"] = base["soot"] * (0.5 + 1.2 * oil_frac) * float(rng.normal(1.0, 0.12))
                row["tbn"] = max(0.4, base["tbn"] * (1.15 - 0.45 * oil_frac) * float(rng.normal(1.0, 0.05)))
                row["water"] = max(0.0, base["water"] * float(rng.lognormal(0, 0.3)))
                row["fuel"] = max(0.0, base["fuel"] * float(rng.lognormal(0, 0.3)))
                row["glycol"] = 0.0

                for ep in episodes:
                    if ep["done"] or not (ep["onset"] <= t < ep["repair"]):
                        continue
                    prog = (t - ep["onset"]) / (ep["repair"] - ep["onset"])
                    ramp = float(prog) ** 1.6
                    for k, m in ep["mult"].items():
                        if k == "visc":
                            row["visc"] *= 1.0 + m * ramp
                        elif k in ("glycol", "water", "fuel"):
                            row[k] = row.get(k, 0.0) + m * ramp
                        else:
                            row[k] = row.get(k, 0.0) + base.get(k, 1.0) * (m - 1.0) * ramp

                for ep in episodes:
                    if not ep["done"] and t >= ep["repair"]:
                        ep["done"] = True
                        wo_seq += 1
                        open_d = ep["repair"] + pd.Timedelta(days=float(rng.uniform(0, 7)))  # reporting lag
                        work_orders.append(dict(
                            WorkOrderId=f"WO{wo_seq:07d}", EquipNum=equip_num, Compartment=comp,
                            WOType="CM", FailureCode=ep["code"], Description=ep["desc"],
                            OpenDate=open_d.normalize().date(),
                            CloseDate=(open_d + pd.Timedelta(days=float(rng.uniform(1, 14)))).normalize().date(),
                            SMUAtWO=round(_smu_at(open_d), 1),
                            DowntimeHours=round(float(rng.uniform(8, 140)), 1),
                            PartsCost=round(float(rng.uniform(2500, 46000)), 2),
                            LabourCost=round(float(rng.uniform(800, 9000)), 2),
                        ))

                sev = _severity(row, fam)
                sample_seq += 1
                report_oil_hours = round(oil_hours, 1) if rng.random() > 0.15 else np.nan
                report_smu = round(smu, 1) if rng.random() > 0.05 else np.nan
                samples.append(dict(
                    SampleNum=390000000 + sample_seq, EquipNum=equip_num, SerialNum=serial,
                    EqpModel=model, ModelFamily=family, Compartment=comp,
                    SiteId=site_id, SiteName=site_name,
                    DateSampled=t.normalize().date(),
                    DateProcessed=(t + pd.Timedelta(days=float(rng.uniform(2, 8)))).normalize().date(),
                    CMeter=report_smu, CMeterFluid=report_oil_hours,
                    FluidChanged=fluid_changed, MakeUpFluid=makeup_l,
                    OverallInterp=_SEV_TO_CODE[sev],
                    InterpText=_interp_text(row, fam, sev),
                    WorkOrderId=np.nan,
                    Fe=round(row["fe"], 1), Cu=round(row["cu"], 1), Cr=round(row["cr"], 2),
                    Pb=round(row["pb"], 1), Al=round(row["al"], 1), Si=round(row["si"], 1),
                    Na=round(row["na"], 1), K=round(row["k"], 1),
                    Water=round(row["water"], 3), Fuel=round(row["fuel"], 2),
                    Glycol=round(row["glycol"], 3), Soot=round(row["soot"], 2),
                    Visc100=round(row["visc"], 2), Oxidation=round(row["oxid"], 1),
                    Nitration=round(row["nitr"], 1), TBN=round(row["tbn"], 2),
                    PQ=round(row["pq"], 1),
                ))

        # a few unrelated corrective jobs per machine
        for _ in range(int(rng.integers(0, 5))):
            d = START + pd.Timedelta(days=float(rng.uniform(0, (END - START).days)))
            wo_seq += 1
            work_orders.append(dict(
                WorkOrderId=f"WO{wo_seq:07d}", EquipNum=equip_num, Compartment="CHASSIS",
                WOType="CM", FailureCode="MSC-GEN", Description="Cab / electrical / body repair",
                OpenDate=d.normalize().date(), CloseDate=(d + pd.Timedelta(days=2)).normalize().date(),
                SMUAtWO=np.nan, DowntimeHours=round(float(rng.uniform(2, 20)), 1),
                PartsCost=round(float(rng.uniform(100, 3000)), 2),
                LabourCost=round(float(rng.uniform(100, 1200)), 2),
            ))

    sos = pd.DataFrame(samples).sort_values(["EquipNum", "Compartment", "DateSampled"]).reset_index(drop=True)
    wo = pd.DataFrame(work_orders).sort_values(["EquipNum", "OpenDate"]).reset_index(drop=True)
    am = pd.DataFrame(assets).sort_values("EquipNum").reset_index(drop=True)
    return {"sos": sos, "work_orders": wo, "assets": am}


def write(seed: int = RNG_SEED) -> dict[str, pd.DataFrame]:
    config.ensure_dirs()
    out = generate(seed)
    out["sos"].to_csv(config.SOS_CSV, index=False)
    out["work_orders"].to_csv(config.WO_CSV, index=False)
    out["assets"].to_csv(config.ASSET_CSV, index=False)
    config.SYNTHETIC_MARKER.write_text(f"generated seed={seed}\n")
    return out


if __name__ == "__main__":
    o = write()
    cm = int((o["work_orders"]["WOType"] == "CM").sum())
    pm = int((o["work_orders"]["WOType"] == "PM").sum())
    print(f"sos_samples.csv : {len(o['sos']):,} rows  "
          f"{o['sos'].DateSampled.min()} -> {o['sos'].DateSampled.max()}")
    print(f"work_orders.csv : {len(o['work_orders']):,} rows ({cm:,} corrective, {pm:,} preventive)")
    print(f"asset_master.csv: {len(o['assets']):,} machines")
