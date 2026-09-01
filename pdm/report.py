"""
pdm/report.py  --  Human-readable, plain-English output formatting.

Every phase calls helpers from here so the terminal output is consistent,
easy to scan, and understandable by anyone (not just data scientists).
"""

from __future__ import annotations

import sys
import io

# Force UTF-8 output on Windows to avoid cp1252 encoding errors with special chars
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def ok(msg: str)   -> str: return f"  [ OK ]  {msg}"
def warn(msg: str) -> str: return f"  [WARN]  {msg}"
def err(msg: str)  -> str: return f"  [FAIL]  {msg}"
def info(msg: str) -> str: return f"  [ .. ]  {msg}"


# ---- Phase 0 -----------------------------------------------------------------
def phase0_summary(machines: int, families: list, compartments: list,
                   fmeca_in: int, fmeca_total: int, horizon: int, gate: str) -> None:
    print()
    print("  WHAT THE PILOT COVERS")
    print(ok(f"Fleet size       : {machines} machines in scope"))
    print(ok(f"Machine types    : {', '.join(families)}"))
    print(ok(f"Parts monitored  : {', '.join(compartments)}"))
    print()
    print("  FAILURE MODES WE CAN DETECT FROM OIL (FMECA)")
    print(ok(f"{fmeca_in} out of {fmeca_total} failure modes are detectable in oil samples"))
    if fmeca_in < fmeca_total:
        print(warn(f"{fmeca_total - fmeca_in} failure mode(s) not detectable via oil -- handled by other means"))
    print()
    print("  WHAT WE ARE PREDICTING")
    print(ok(f"Goal : Will this machine-compartment need a corrective repair within {horizon} days?"))
    gate_tag = "[ OK ]" if gate.startswith("PASS") else "[FAIL]"
    print(f"  {gate_tag}  Gate : {gate}")


# ---- Phase 1 -----------------------------------------------------------------
def phase1_summary(n_samples: int, reversals: int,
                   wo_map_pct: float | None, cm_share: float | None,
                   lag_median: float | None, oil_hrs_pct: float,
                   oil_rating: str, no_wo: bool, stop_fix: bool,
                   stop_reason: str | None) -> None:
    print()
    print("  DATA LOADED & CLEANED")
    print(ok(f"{n_samples:,} oil samples loaded and standardised"))
    if reversals == 0:
        print(ok("No hour-meter reversals found (odometer readings look clean)"))
    else:
        print(warn(f"{reversals} hour-meter reversals found -- check these machines"))

    print()
    print("  WORK ORDER QUALITY")
    if no_wo:
        print(warn("No work order file found -- Phase 3 ML model will be skipped until you provide one"))
    elif wo_map_pct is not None:
        tag = "[ OK ]" if wo_map_pct >= 0.50 else "[FAIL]"
        print(f"  {tag}  Work orders linked to the right machine part : {wo_map_pct:.0%}  "
              f"(need at least 50%)")
        if cm_share is not None:
            print(ok(f"Corrective (breakdown) work orders : {cm_share:.0%} of all WOs"))
        if lag_median is not None:
            tag2 = "[ OK ]" if lag_median <= 60 else "[WARN]"
            print(f"  {tag2}  Median delay between oil alarm and work order raised : {lag_median:.0f} days")

    print()
    print("  OIL SAMPLE COMPLETENESS")
    tag3 = "[ OK ]" if oil_rating == "GOOD" else "[WARN]"
    note = "(good enough to detect wear trends)" if oil_rating == "GOOD" \
           else "(below 70% target -- request better data)"
    print(f"  {tag3}  {oil_rating}  Oil hours recorded on {oil_hrs_pct:.0%} of samples  {note}")

    if stop_fix:
        print()
        print(err(f"STOP -- DO NOT run Phase 3 yet:  {stop_reason}"))
        print(err("Fix the work order data first, then re-run the pipeline."))


# ---- Phase 2 -----------------------------------------------------------------
def phase2_summary(n_limits: int, n_tighter: int, n_watchlist: int,
                   no_chemistry: bool, top_rows: list[dict]) -> None:
    print()
    print("  FLEET-SPECIFIC ALARM LIMITS (ASTM D7720)")
    print(ok(f"Calculated {n_limits} alarm limits from your own fleet history"))
    if n_tighter > 0:
        print(warn(f"{n_tighter} of those limits are TIGHTER than the generic OEM handbook limits"))
        print(warn("  This means your fleet wears faster than average; generic OEM limits alone are too lenient"))
    else:
        print(ok("All fleet limits are within the OEM handbook range -- wear rates look normal"))

    print()
    print("  WEEKLY WATCHLIST -- MACHINES THAT NEED ATTENTION")
    print(ok(f"{n_watchlist} machine-compartment combinations monitored this week"))
    if no_chemistry:
        print(warn("No numeric lab results available -- ranking uses lab severity text only (less precise)"))

    print()
    print(f"  {'#':<4} {'Machine':<14} {'Part':<12} {'Severity':<10} {'Score':<8}  What the score means")
    print(f"  {'-'*4} {'-'*14} {'-'*12} {'-'*10} {'-'*8}  {'-'*40}")
    for r in top_rows:
        sev     = r['lab_severity']
        score   = r['priority_score']
        novelty = r['novelty_z']
        exceed  = int(r['d7720_action_exceedances'])
        explain = f"{novelty:+.1f} sigma outside normal cluster"
        if exceed > 0:
            explain += f", exceeded alarm limit {exceed}x"
        sev_tag = "[!!!]" if sev == "ACTION" else ("[!] " if sev == "MONITOR" else "[ ] ")
        print(f"  {r['rank']:>2}.  {r['machine_id']:<14} {r['component']:<12} "
              f"{sev_tag} {sev:<9} {score:>6.0f}   {explain}")

    print()
    print(info("Score = lab severity + distance from normal cluster + how many alarm limits exceeded"))
    print(info("Higher score = more urgent.  [!!!] ACTION = lab says inspect now.  [!] MONITOR = keep watching."))


# ---- Phase 3 -----------------------------------------------------------------
def phase3_summary(horizon_sweep: list[dict], best_name: str, best_prauc: float,
                   rule_prauc: float, base_rate: float,
                   precision_cap: float, recall_cap: float,
                   ship_rule: bool) -> None:
    print()
    print("  HOW FAR AHEAD CAN WE PREDICT? (HORIZON SWEEP)")
    print(f"  {'Predict window':<18} {'Failures found':<18}  What it means")
    print(f"  {'-'*18} {'-'*18}  {'-'*50}")
    for row in horizon_sweep:
        d    = row['horizon_days']
        p    = row['positives']
        note = {
            14: "Very short -- only urgent alarms",
            30: "<-- SELECTED: best balance of time to act vs. accuracy",
            60: "More failures found, but predictions less precise",
            90: "Most failures caught, but 3-month window is too vague for planning",
        }.get(d, "")
        print(f"  Within {d:>2} days  :  {p:>4} failure events     {note}")

    print()
    print("  HOW ACCURATE IS THE MODEL?")
    lift = best_prauc / max(base_rate, 0.001)
    if ship_rule:
        print(warn("The ML model did NOT beat the simple rule-based alarm on the test window"))
        print(warn("Using Phase 2 rule alarm as output -- gather more failure data and re-run"))
    else:
        tag = "[ OK ]" if best_prauc >= 0.50 else "[WARN]"
        print(f"  {tag}  Model type          : {best_name}")
        print(f"  {tag}  Accuracy (PR-AUC)   : {best_prauc:.3f}  "
              f"(vs simple alarm rule : {rule_prauc:.3f}  |  random guess : {base_rate:.3f})")
        print(f"  {tag}  The model is {lift:.0f}x better than random guessing")
        print()
        print("  WHAT THIS MEANS FOR INSPECTION CAPACITY")
        print(info(f"If you inspect the top-ranked machines : you will catch {recall_cap:.0%} of all coming failures"))
        print(info(f"Of those inspections : {precision_cap:.0%} will actually find a real problem"))
        if precision_cap < 0.30:
            print(warn("Precision is low -- expected with small fleets. Grows as more data accumulates."))


# ---- Phase 4 -----------------------------------------------------------------
def phase4_summary(n_rows: int, n_events: int, prauc_30d: float,
                   lift_30d: float, km_30: float, km_60: float, km_90: float,
                   n_scored: int, top_rows: list[dict]) -> None:
    print()
    print("  FLEET BASELINE -- HOW OFTEN DO MACHINES NORMALLY FAIL?")
    print(ok("Background failure rate for a healthy machine in this fleet:"))
    print(ok(f"  Within 30 days : {(1-km_30)*100:.0f}%   "
             f"Within 60 days : {(1-km_60)*100:.0f}%   "
             f"Within 90 days : {(1-km_90)*100:.0f}%"))

    print()
    print("  HOW WELL DOES THE SURVIVAL MODEL PREDICT FAILURES?")
    tag = "[ OK ]" if lift_30d >= 5 else "[WARN]"
    print(f"  {tag}  30-day risk prediction is {lift_30d:.0f}x better than random guessing")
    print(f"  {tag}  Trained on {n_rows:,} observations, {n_events} actual failure events")

    print()
    print("  MACHINES WITH HIGHEST FAILURE RISK RIGHT NOW")
    print(f"  {'Machine':<14} {'Part':<12} {'Lab':<9} {'Urgency':<7} {'30-day':<10} {'60-day':<10} {'90-day'}")
    print(f"  {'-'*14} {'-'*12} {'-'*9} {'-'*7} {'-'*10} {'-'*10} {'-'*8}")
    for r in top_rows:
        r30  = r['risk_30d']
        tag2 = "[!!!]" if r30 > 0.7 else ("[!] " if r30 > 0.4 else "[ ] ")
        print(f"  {r['machine_id']:<14} {r['component']:<12} {r['lab_severity']:<9} "
              f"{tag2}   {r30:.0%}      {r['risk_60d']:.0%}      {r['risk_90d']:.0%}")
    print()
    print(info(f"{n_scored} machine-compartments scored  -->  artifacts/phase4_multi_horizon_risk.csv"))


# ---- Phase 5 -----------------------------------------------------------------
def phase5_summary(risk_source: str, n_total: int, n_p1: int, n_p2: int, n_p3: int,
                   prevented_usd: float, inspection_cost_usd: float,
                   net_value_usd: float, roi: float,
                   top_rows: list[dict]) -> None:
    print()
    print("  WHAT DOES EACH PRIORITY LEVEL MEAN?")
    print("  [P1 IMMEDIATE]  Schedule a shutdown inspection before the next shift if possible")
    print("  [P2 PLAN]       Add to the next scheduled service interval")
    print("  [P3 MONITOR]    No action yet -- keep taking oil samples on schedule")

    print()
    print("  THIS WEEK'S INSPECTION WORKLIST")
    print(f"  P1 Immediate inspections : {n_p1}   |   "
          f"P2 Plan at next service : {n_p2}   |   "
          f"P3 Monitor only : {n_p3}")
    print()

    print(f"  {'#':<4} {'Machine':<14} {'Part':<12} {'Priority':<12} {'Risk':<8} {'Expected cost saved'}")
    print(f"  {'-'*4} {'-'*14} {'-'*12} {'-'*12} {'-'*8} {'-'*22}")
    for r in top_rows:
        tier = r['tier']
        risk = r['risk']
        tag  = "[P1 IMMEDIATE]" if tier == "P1" else ("[P2 PLAN]     " if tier == "P2" else "[P3 MONITOR]  ")
        print(f"  {r['rank']:>2}.  {r['machine_id']:<14} {r['component']:<12} "
              f"{tag}  {risk:>5.0%}    ${r['expected_value_usd']:>16,.0f}")

    print()
    print("  BUSINESS CASE (THIS WEEK)")
    print(ok(f"Total expected failure cost avoided : ${prevented_usd:,.0f}"))
    print(ok(f"Total inspection cost               : ${inspection_cost_usd:,.0f}"))
    print(ok(f"Net expected value saved            : ${net_value_usd:,.0f}"))
    tag_roi = "[ OK ]" if roi >= 5 else "[WARN]"
    print(f"  {tag_roi}  Return on investment : {roi:.1f}x   "
          f"(every $1 spent on inspections saves ${roi:.1f} in avoided breakdowns)")
    print()
    print(info("Full ranked list      -->  data/processed/phase5_cost_ranked_advisories.csv"))
    print(info("After each inspection, fill in 'inspection_outcome' in artifacts/advisory_feedback_table.csv"))
    print(info("Six months of feedback builds clean training labels for Phase 3 re-train"))
