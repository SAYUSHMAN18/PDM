"""
Generate realistic-looking S.O.S (oil analysis) and Work Order data.

Purpose: let you build and test the whole pipeline TODAY, before your real
extracts arrive. When the real files land, delete this script and point
build_dataset.py at your own CSVs -- the column names below are the contract.

Output:
    data/raw/sos_samples.csv
    data/raw/work_orders.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2026-06-30")

FLEET = (
    [(f"EX-{i:03d}", "EXCAVATOR", "336") for i in range(1, 21)]
    + [(f"WL-{i:03d}", "WHEEL_LOADER", "966") for i in range(1, 16)]
    + [(f"HT-{i:03d}", "HAUL_TRUCK", "775") for i in range(1, 16)]
)

# Baseline chemistry per component (healthy machine, fresh-ish oil).
BASE = {
    "ENGINE": dict(fe=18, cu=7, cr=2.0, pb=4, al=5, si=8, na=4, k=3,
                   visc=108, oxid=12, nitr=8, soot=0.4, tbn=9.0,
                   water=0.02, fuel=0.6, glycol=0.0, pq=8,
                   oil_change_h=500),
    "HYDRAULIC": dict(fe=11, cu=4, cr=1.0, pb=2, al=3, si=6, na=3, k=2,
                      visc=46, oxid=8, nitr=4, soot=0.05, tbn=4.0,
                      water=0.03, fuel=0.0, glycol=0.0, pq=6,
                      oil_change_h=2000),
    "TRANSMISSION": dict(fe=38, cu=14, cr=3.0, pb=6, al=5, si=7, na=3, k=2,
                         visc=68, oxid=9, nitr=5, soot=0.1, tbn=6.0,
                         water=0.03, fuel=0.0, glycol=0.0, pq=14,
                         oil_change_h=1000),
    "FINAL_DRIVE": dict(fe=85, cu=9, cr=5.0, pb=8, al=6, si=9, na=3, k=2,
                        visc=100, oxid=7, nitr=4, soot=0.05, tbn=5.0,
                        water=0.05, fuel=0.0, glycol=0.0, pq=30,
                        oil_change_h=2000),
}

# Fault archetypes: which measurements climb, and which repair they end in.
FAULTS = {
    "ENGINE": [
        ("BEARING_WEAR", dict(pb=6.0, cu=4.0, fe=2.2, pq=3.0), "ENG-BRG", "Main bearing wear"),
        ("DIRT_INGRESS", dict(si=7.0, al=3.5, fe=2.0), "ENG-AIR", "Air induction leak / dirt ingress"),
        ("COOLANT_LEAK", dict(glycol=1.0, na=6.0, k=6.0, water=8.0), "ENG-COOL", "Coolant entering crankcase"),
        ("FUEL_DILUTION", dict(fuel=6.0, visc=-0.28, soot=2.0), "ENG-INJ", "Injector leak / fuel dilution"),
    ],
    "HYDRAULIC": [
        ("PUMP_WEAR", dict(fe=3.2, cu=3.5, cr=2.5, pq=4.0), "HYD-PMP", "Hydraulic pump wear"),
        ("DIRT_INGRESS", dict(si=8.0, fe=2.4, al=3.0), "HYD-SEAL", "Cylinder seal / dirt ingress"),
        ("WATER_INGRESS", dict(water=12.0, fe=2.0), "HYD-WTR", "Water contamination"),
    ],
    "TRANSMISSION": [
        ("GEAR_WEAR", dict(fe=3.0, cr=3.0, pq=4.5), "TRN-GER", "Gear / clutch pack wear"),
        ("BEARING_WEAR", dict(cu=4.0, pb=5.0, fe=2.2), "TRN-BRG", "Transmission bearing wear"),
    ],
    "FINAL_DRIVE": [
        ("GEAR_WEAR", dict(fe=3.0, cr=3.5, pq=5.0), "FDR-GER", "Final drive gear wear"),
        ("SEAL_LEAK", dict(si=6.0, water=8.0, fe=1.8), "FDR-SEAL", "Duo-cone seal leak"),
    ],
}

COMPONENTS = list(BASE)


def _lab_flag(row: dict, comp: str) -> str:
    """Crude stand-in for the lab's own severity call."""
    b = BASE[comp]
    score = 0
    for k in ("fe", "cu", "cr", "pb", "al", "si", "pq"):
        r = row[k] / max(b[k], 1e-6)
        score += 2 if r > 3 else 1 if r > 1.8 else 0
    if row["water"] > 0.5:
        score += 3
    if row["glycol"] > 0.1:
        score += 4
    if row["fuel"] > 5:
        score += 2
    if abs(row["visc"] - b["visc"]) / b["visc"] > 0.15:
        score += 2
    return "CRITICAL" if score >= 8 else "ACTION" if score >= 5 else "MONITOR" if score >= 2 else "NORMAL"


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    samples, work_orders = [], []
    wo_seq = 0

    for machine_id, mtype, model in FLEET:
        hours_per_day = float(np.clip(RNG.normal(9.0, 2.0), 4.0, 14.0))
        smu0 = float(RNG.uniform(500, 9000))          # service meter at t0

        for comp in COMPONENTS:
            oil_change_h = BASE[comp]["oil_change_h"]
            interval_days = oil_change_h / hours_per_day / 2.0   # sample ~2x per oil life
            oil_hours = float(RNG.uniform(0, oil_change_h))
            t = START + pd.Timedelta(days=float(RNG.uniform(0, 25)))
            smu = smu0

            # Schedule 0-3 fault episodes for this machine-component.
            episodes = []
            for _ in range(RNG.integers(0, 4)):
                onset = START + pd.Timedelta(days=float(RNG.uniform(30, (END - START).days - 40)))
                dur = float(RNG.uniform(45, 160))       # days from onset to repair
                name, mult, code, desc = FAULTS[comp][RNG.integers(len(FAULTS[comp]))]
                episodes.append(dict(onset=onset, repair=onset + pd.Timedelta(days=dur),
                                     mult=mult, code=code, desc=desc, name=name, done=False))
            episodes.sort(key=lambda e: e["onset"])

            while t < END:
                step_days = float(np.clip(RNG.normal(interval_days, interval_days * 0.18), 7, 90))
                t += pd.Timedelta(days=step_days)
                if t >= END:
                    break
                smu += step_days * hours_per_day
                oil_hours += step_days * hours_per_day

                # Oil change resets oil hours (and the PM work order that goes with it).
                if oil_hours >= oil_change_h:
                    oil_hours = float(RNG.uniform(0, 60))
                    wo_seq += 1
                    work_orders.append(dict(
                        wo_id=f"WO{wo_seq:07d}", machine_id=machine_id, component=comp,
                        wo_type="PM", failure_code="PM-OIL",
                        description=f"Scheduled {comp.lower()} oil and filter change",
                        open_date=t.normalize(), close_date=(t + pd.Timedelta(days=1)).normalize(),
                        smu_at_wo=round(smu, 1), downtime_hours=round(float(RNG.uniform(2, 6)), 1),
                        parts_cost=round(float(RNG.uniform(150, 900)), 2),
                        labour_cost=round(float(RNG.uniform(100, 400)), 2),
                    ))

                b = BASE[comp]
                oil_frac = oil_hours / oil_change_h          # wear metals accumulate with oil age
                row = {}
                for k in ("fe", "cu", "cr", "pb", "al", "si", "na", "k", "pq"):
                    row[k] = b[k] * (0.55 + 0.9 * oil_frac) * float(RNG.lognormal(0, 0.16))
                row["visc"] = b["visc"] * float(RNG.normal(1.0, 0.025))
                row["oxid"] = b["oxid"] * (0.7 + 0.7 * oil_frac) * float(RNG.normal(1.0, 0.08))
                row["nitr"] = b["nitr"] * (0.7 + 0.7 * oil_frac) * float(RNG.normal(1.0, 0.08))
                row["soot"] = b["soot"] * (0.5 + 1.2 * oil_frac) * float(RNG.normal(1.0, 0.12))
                row["tbn"] = max(0.5, b["tbn"] * (1.15 - 0.45 * oil_frac) * float(RNG.normal(1.0, 0.05)))
                row["water"] = max(0.0, b["water"] * float(RNG.lognormal(0, 0.3)))
                row["fuel"] = max(0.0, b["fuel"] * float(RNG.lognormal(0, 0.3)))
                row["glycol"] = 0.0

                # Apply any active fault episode: severity ramps toward the repair date.
                for ep in episodes:
                    if ep["done"] or not (ep["onset"] <= t < ep["repair"]):
                        continue
                    prog = (t - ep["onset"]) / (ep["repair"] - ep["onset"])   # 0 -> 1
                    ramp = prog ** 1.6
                    for k, m in ep["mult"].items():
                        if m < 0:                       # negative = drives the value down (e.g. viscosity)
                            row[k] *= 1.0 + m * ramp
                        else:
                            row[k] = row[k] + b.get(k, 1.0) * (m - 1.0) * ramp + (m * ramp if k in ("glycol", "water", "fuel") else 0)

                # Repair happened -> corrective work order, chemistry returns to normal.
                for ep in episodes:
                    if not ep["done"] and t >= ep["repair"]:
                        ep["done"] = True
                        wo_seq += 1
                        open_d = ep["repair"] + pd.Timedelta(days=float(RNG.uniform(0, 6)))  # reporting lag
                        work_orders.append(dict(
                            wo_id=f"WO{wo_seq:07d}", machine_id=machine_id, component=comp,
                            wo_type="CM", failure_code=ep["code"], description=ep["desc"],
                            open_date=open_d.normalize(),
                            close_date=(open_d + pd.Timedelta(days=float(RNG.uniform(1, 12)))).normalize(),
                            smu_at_wo=round(smu, 1),
                            downtime_hours=round(float(RNG.uniform(8, 120)), 1),
                            parts_cost=round(float(RNG.uniform(1500, 45000)), 2),
                            labour_cost=round(float(RNG.uniform(600, 9000)), 2),
                        ))

                flag = _lab_flag(row, comp)
                samples.append(dict(
                    sample_id=f"S{len(samples)+1:08d}", machine_id=machine_id,
                    machine_type=mtype, model=model, component=comp,
                    sample_date=t.normalize(), smu_hours=round(smu, 1),
                    oil_hours=round(oil_hours, 1), oil_changed="N",
                    fe_ppm=round(row["fe"], 1), cu_ppm=round(row["cu"], 1),
                    cr_ppm=round(row["cr"], 2), pb_ppm=round(row["pb"], 1),
                    al_ppm=round(row["al"], 1), si_ppm=round(row["si"], 1),
                    na_ppm=round(row["na"], 1), k_ppm=round(row["k"], 1),
                    water_pct=round(row["water"], 3), fuel_pct=round(row["fuel"], 2),
                    glycol_pct=round(row["glycol"], 3), soot_pct=round(row["soot"], 2),
                    visc40=round(row["visc"], 1), oxidation=round(row["oxid"], 1),
                    nitration=round(row["nitr"], 1), tbn=round(row["tbn"], 2),
                    pq_index=round(row["pq"], 1), lab_severity=flag,
                ))

        # A few unrelated corrective jobs, so the CMMS isn't unrealistically tidy.
        for _ in range(RNG.integers(0, 5)):
            d = START + pd.Timedelta(days=float(RNG.uniform(0, (END - START).days)))
            wo_seq += 1
            work_orders.append(dict(
                wo_id=f"WO{wo_seq:07d}", machine_id=machine_id, component="OTHER",
                wo_type="CM", failure_code="MSC-GEN", description="Cab / electrical / body repair",
                open_date=d.normalize(), close_date=(d + pd.Timedelta(days=2)).normalize(),
                smu_at_wo=np.nan, downtime_hours=round(float(RNG.uniform(2, 20)), 1),
                parts_cost=round(float(RNG.uniform(100, 3000)), 2),
                labour_cost=round(float(RNG.uniform(100, 1200)), 2),
            ))

    sos = pd.DataFrame(samples).sort_values(["machine_id", "component", "sample_date"])
    wo = pd.DataFrame(work_orders).sort_values(["machine_id", "open_date"])
    return sos, wo


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    sos, wo = generate()
    sos.to_csv("data/raw/sos_samples.csv", index=False)
    wo.to_csv("data/raw/work_orders.csv", index=False)
    print(f"sos_samples.csv : {len(sos):,} rows, {sos.sample_date.min().date()} -> {sos.sample_date.max().date()}")
    print(f"work_orders.csv : {len(wo):,} rows "
          f"({(wo.wo_type == 'CM').sum():,} corrective, {(wo.wo_type == 'PM').sum():,} preventive)")
