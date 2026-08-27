"""
Central configuration: paths, taxonomies, the cost model and the pilot scope.

Nothing here reads data. Every other module imports its constants from here so
there is exactly one place to change a threshold, a sump volume or a cost.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

# Filenames the loader looks for. Drop real extracts here with these names and
# the pipeline runs on them with no code change (see data.py).
SOS_CSV = DATA_RAW / "sos_samples.csv"
WO_CSV = DATA_RAW / "work_orders.csv"
ASSET_CSV = DATA_RAW / "asset_master.csv"
SYNTHETIC_MARKER = DATA_RAW / ".synthetic"          # written by synth.py
SOS_XLSX = DATA_RAW / "SosFluidSample.xlsx"          # the 8-row real sample extract

# --------------------------------------------------------------------- analytes
# Internal canonical analyte names. The loader maps whatever the source calls
# them onto these.
ANALYTES = [
    "fe_ppm", "cu_ppm", "cr_ppm", "pb_ppm", "al_ppm", "si_ppm", "na_ppm", "k_ppm",
    "water_pct", "fuel_pct", "glycol_pct", "soot_pct",
    "visc100", "oxidation", "nitration", "tbn", "pq_index",
]
WEAR_METALS = ["fe_ppm", "cu_ppm", "cr_ppm", "pb_ppm", "al_ppm", "si_ppm"]

# Analytes where "more" unambiguously means "worse" -- used for monotonic model
# constraints and for one-sided alarm limits.
HIGHER_IS_WORSE = [
    "fe_ppm", "cu_ppm", "cr_ppm", "pb_ppm", "al_ppm", "si_ppm", "na_ppm", "k_ppm",
    "water_pct", "fuel_pct", "glycol_pct", "soot_pct",
    "oxidation", "nitration", "pq_index",
]

ANALYTE_PRETTY = {
    "fe_ppm": "iron", "cu_ppm": "copper", "cr_ppm": "chromium", "pb_ppm": "lead",
    "al_ppm": "aluminium", "si_ppm": "silicon (dirt)", "na_ppm": "sodium",
    "k_ppm": "potassium", "water_pct": "water", "fuel_pct": "fuel dilution",
    "glycol_pct": "glycol (coolant)", "soot_pct": "soot",
    "visc100": "viscosity @100C", "oxidation": "oxidation",
    "nitration": "nitration", "tbn": "TBN (additive reserve)",
    "pq_index": "PQ (ferrous debris)",
}

# --------------------------------------------------------------------- severity
SEVERITY_ORDER = {"NORMAL": 0, "MONITOR": 1, "ACTION": 2, "CRITICAL": 3}
HEALTHY_SEVERITIES = ("NORMAL", "MONITOR")   # population used to fit baselines

# Real lab "OverallInterp" codes -> canonical severity. U = unavailable/no call.
INTERP_CODE_MAP = {
    "A": "NORMAL", "N": "NORMAL",
    "B": "MONITOR", "S": "MONITOR", "C": "MONITOR", "M": "MONITOR",
    "AR": "ACTION", "R": "ACTION",
    "CR": "CRITICAL", "X": "CRITICAL",
    "U": "NORMAL", "": "NORMAL", "NAN": "NORMAL",
}

# --------------------------------------------------------------------- horizons
HORIZONS = [14, 30, 60, 90]
PRIMARY_HORIZON = 30           # "corrective WO within N days of the sample?"
SURVIVAL_HORIZONS = [30, 60, 90]  # multi-horizon risk reported by Phase 4
SURVIVAL_CENSOR_D = 90        # administrative censoring horizon for the survival fit
SURVIVAL_STEP_D = 15         # width of each discrete hazard period
POST_REPAIR_BLACKOUT_D = 10    # ignore samples this soon after a repair
PEER_TRAIN_FRACTION = 0.70     # earliest N of the timeline builds peer baselines
TIME_SPLIT_FRACTION = 0.70     # earliest N trains, latest 1-N tests (never random)
INSPECTION_CAPACITY_FRAC = 0.15   # share of the fleet an engineer can inspect / cycle

# --------------------------------------------------------- compartment taxonomy
# Compartment code -> component family, physical position and typical oil sump
# volume (litres). Volumes drive the ppm -> wear-mass conversion in features.py.
COMPARTMENT_TAXONOMY = {
    "ENGINE":       {"family": "ENGINE",       "position": "PRIMARY",     "sump_volume_l": 60.0},
    "TRANSMISSION": {"family": "TRANSMISSION", "position": "PRIMARY",     "sump_volume_l": 110.0},
    "TRANS":        {"family": "TRANSMISSION", "position": "PRIMARY",     "sump_volume_l": 110.0},
    "HYDRAULIC":    {"family": "HYDRAULIC",    "position": "PRIMARY",     "sump_volume_l": 150.0},
    "HYD":          {"family": "HYDRAULIC",    "position": "PRIMARY",     "sump_volume_l": 150.0},
    "DIFF_FR":      {"family": "DIFFERENTIAL", "position": "FRONT",       "sump_volume_l": 85.0},
    "DIFF_RR":      {"family": "DIFFERENTIAL", "position": "REAR",        "sump_volume_l": 85.0},
    "DIFF":         {"family": "DIFFERENTIAL", "position": "PRIMARY",     "sump_volume_l": 85.0},
    "FD_FR_LT":     {"family": "FINAL_DRIVE",  "position": "FRONT_LEFT",  "sump_volume_l": 45.0},
    "FD_FR_RT":     {"family": "FINAL_DRIVE",  "position": "FRONT_RIGHT", "sump_volume_l": 45.0},
    "FD_RR_LT":     {"family": "FINAL_DRIVE",  "position": "REAR_LEFT",   "sump_volume_l": 45.0},
    "FD_RR_RT":     {"family": "FINAL_DRIVE",  "position": "REAR_RIGHT",  "sump_volume_l": 45.0},
    "FD_LT":        {"family": "FINAL_DRIVE",  "position": "LEFT",        "sump_volume_l": 40.0},
    "FD_RT":        {"family": "FINAL_DRIVE",  "position": "RIGHT",       "sump_volume_l": 40.0},
    "WH_FR_LT":     {"family": "WHEEL_END",    "position": "FRONT_LEFT",  "sump_volume_l": 25.0},
    "WH_FR_RT":     {"family": "WHEEL_END",    "position": "FRONT_RIGHT", "sump_volume_l": 25.0},
    "WH_RR_LT":     {"family": "WHEEL_END",    "position": "REAR_LEFT",   "sump_volume_l": 25.0},
    "WH_RR_RT":     {"family": "WHEEL_END",    "position": "REAR_RIGHT",  "sump_volume_l": 25.0},
    "SWING":        {"family": "SWING_DRIVE",  "position": "PRIMARY",     "sump_volume_l": 18.0},
}
DEFAULT_SUMP_VOLUME_L = 50.0

def compartment_family(code: str) -> str:
    return COMPARTMENT_TAXONOMY.get(str(code).upper().strip(), {}).get("family", "OTHER")

def compartment_sump_l(code: str) -> float:
    return COMPARTMENT_TAXONOMY.get(str(code).upper().strip(), {}).get(
        "sump_volume_l", DEFAULT_SUMP_VOLUME_L)

def compartment_position(code: str) -> str:
    return COMPARTMENT_TAXONOMY.get(str(code).upper().strip(), {}).get("position", "UNKNOWN")

# ------------------------------------------------- generic OEM condemning limits
# Universal "caution" levels (ppm / % ) by component family. Phase 2 derives
# fleet-relative D7720 limits and reports where they land tighter than these.
GENERIC_CONDEMNING_LIMITS = {
    "ENGINE":       {"fe_ppm": 100, "cu_ppm": 40,  "cr_ppm": 15, "pb_ppm": 30, "al_ppm": 25, "si_ppm": 25, "water_pct": 0.2,  "glycol_pct": 0.1, "fuel_pct": 5.0, "soot_pct": 3.0, "pq_index": 50},
    "TRANSMISSION": {"fe_ppm": 150, "cu_ppm": 60,  "cr_ppm": 20, "pb_ppm": 40, "al_ppm": 25, "si_ppm": 25, "water_pct": 0.2,  "pq_index": 80},
    "HYDRAULIC":    {"fe_ppm": 50,  "cu_ppm": 25,  "cr_ppm": 10, "pb_ppm": 20, "al_ppm": 15, "si_ppm": 20, "water_pct": 0.1,  "pq_index": 30},
    "DIFFERENTIAL": {"fe_ppm": 300, "cu_ppm": 50,  "cr_ppm": 40, "pb_ppm": 50, "al_ppm": 30, "si_ppm": 30, "water_pct": 0.3,  "pq_index": 150},
    "FINAL_DRIVE":  {"fe_ppm": 250, "cu_ppm": 40,  "cr_ppm": 35, "pb_ppm": 40, "al_ppm": 30, "si_ppm": 30, "water_pct": 0.3,  "pq_index": 120},
    "WHEEL_END":    {"fe_ppm": 200, "cu_ppm": 35,  "cr_ppm": 30, "pb_ppm": 35, "al_ppm": 25, "si_ppm": 25, "water_pct": 0.3,  "pq_index": 100},
    "SWING_DRIVE":  {"fe_ppm": 150, "cu_ppm": 35,  "cr_ppm": 25, "pb_ppm": 30, "al_ppm": 25, "si_ppm": 25, "water_pct": 0.3,  "pq_index": 90},
}

# --------------------------------------------------------------- failure cost model
# Fallback economics per component family, used when the work-order history has
# too few closed corrective jobs to estimate a real median (see phase5).
FAILURE_COST = {
    "ENGINE":       {"parts_usd": 35000, "labour_usd": 8000, "downtime_h": 60, "downtime_usd_per_h": 300},
    "TRANSMISSION": {"parts_usd": 22000, "labour_usd": 5500, "downtime_h": 44, "downtime_usd_per_h": 300},
    "HYDRAULIC":    {"parts_usd": 14000, "labour_usd": 4000, "downtime_h": 28, "downtime_usd_per_h": 300},
    "DIFFERENTIAL": {"parts_usd": 18000, "labour_usd": 4500, "downtime_h": 36, "downtime_usd_per_h": 300},
    "FINAL_DRIVE":  {"parts_usd": 12000, "labour_usd": 3200, "downtime_h": 24, "downtime_usd_per_h": 300},
    "WHEEL_END":    {"parts_usd": 8000,  "labour_usd": 2200, "downtime_h": 18, "downtime_usd_per_h": 300},
    "SWING_DRIVE":  {"parts_usd": 10000, "labour_usd": 2800, "downtime_h": 20, "downtime_usd_per_h": 300},
    "OTHER":        {"parts_usd": 6000,  "labour_usd": 1800, "downtime_h": 12, "downtime_usd_per_h": 300},
}
INSPECTION_COST_USD = 350          # cost of acting on one alert (labour + resample)
DETECTION_RATE = 0.60              # share of true problems an inspection actually catches in time
P1_EXPECTED_VALUE_USD = 12000      # expected-cost-at-risk thresholds for the worklist
P2_EXPECTED_VALUE_USD = 4000

# ------------------------------------------------------------- FMECA pilot scope
# The shortlist Phase 0 signs off: (component family x failure mode) pairs where
# oil analysis genuinely carries the signal. Ranked by severity x cost, not RPN.
FMECA_SHORTLIST = [
    {"component_family": "ENGINE",       "failure_mode": "Main / con-rod bearing wear",
     "oil_detectability": "HIGH", "primary_analytes": ["pb_ppm", "cu_ppm", "fe_ppm", "pq_index"],
     "severity_rank": 1, "typical_failure_cost_usd": 45000},
    {"component_family": "ENGINE",       "failure_mode": "Coolant leak into crankcase",
     "oil_detectability": "HIGH", "primary_analytes": ["glycol_pct", "na_ppm", "k_ppm", "water_pct"],
     "severity_rank": 2, "typical_failure_cost_usd": 38000},
    {"component_family": "ENGINE",       "failure_mode": "Injector leak / fuel dilution",
     "oil_detectability": "MEDIUM", "primary_analytes": ["fuel_pct", "visc100", "soot_pct"],
     "severity_rank": 6, "typical_failure_cost_usd": 12000},
    {"component_family": "TRANSMISSION", "failure_mode": "Clutch pack / gear wear",
     "oil_detectability": "HIGH", "primary_analytes": ["fe_ppm", "cr_ppm", "pq_index"],
     "severity_rank": 4, "typical_failure_cost_usd": 26000},
    {"component_family": "HYDRAULIC",    "failure_mode": "Pump wear",
     "oil_detectability": "HIGH", "primary_analytes": ["fe_ppm", "cu_ppm", "cr_ppm", "pq_index"],
     "severity_rank": 5, "typical_failure_cost_usd": 16000},
    {"component_family": "HYDRAULIC",    "failure_mode": "Water / dirt ingress past seals",
     "oil_detectability": "HIGH", "primary_analytes": ["water_pct", "si_ppm", "fe_ppm"],
     "severity_rank": 7, "typical_failure_cost_usd": 9000},
    {"component_family": "DIFFERENTIAL", "failure_mode": "Crown & pinion scuffing",
     "oil_detectability": "HIGH", "primary_analytes": ["fe_ppm", "cr_ppm", "pq_index"],
     "severity_rank": 3, "typical_failure_cost_usd": 28000},
    {"component_family": "FINAL_DRIVE",  "failure_mode": "Duo-cone seal leak / dirt ingress",
     "oil_detectability": "HIGH", "primary_analytes": ["si_ppm", "al_ppm", "water_pct", "fe_ppm"],
     "severity_rank": 8, "typical_failure_cost_usd": 22000},
    {"component_family": "WHEEL_END",    "failure_mode": "Planetary gear / bearing spalling",
     "oil_detectability": "HIGH", "primary_analytes": ["fe_ppm", "cu_ppm", "pq_index"],
     "severity_rank": 9, "typical_failure_cost_usd": 18000},
]

# Families the model is NOT scoped to predict from oil (FMECA "oil cannot see it").
OUT_OF_SCOPE_NOTE = (
    "Structural cracks, hose bursts, electrical and hydraulic-actuator faults are "
    "excluded: oil chemistry does not carry those signals."
)


def ensure_dirs() -> None:
    for d in (DATA_RAW, DATA_PROCESSED, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)
