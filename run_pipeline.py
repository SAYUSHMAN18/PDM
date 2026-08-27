"""
Master Pipeline Runner — Oil Analysis Predictive Maintenance Pilot

Runs the complete 6-phase predictive maintenance pipeline in sequence:
  Phase 0: Scope & FMECA Definition
  Phase 1: Data Foundation, Label Audit & Physics Wear Normalization
  Phase 2: State Detection (ASTM D7720 Limits, EWMA, Novelty Score, Watchlist) — NO WOs Needed
  Phase 3: Supervised Health Assessment (Horizon Sweep, Leakage Guards, ML Model Calibration)
  Phase 4: Prognostics (Multi-Horizon Survival Risk: 30d, 60d, 90d)
  Phase 5: Advisory Generation & Deployment (Cost-Weighted Expected Risk Ranking)

Usage:
  python run_pipeline.py              # Runs all 6 phases end-to-end
  python run_pipeline.py --phase 2    # Runs a specific phase
"""

from __future__ import annotations

import sys
import argparse

import phase0_scope
import phase1_foundation
import phase2_state_detection
import phase3_supervised_model
import phase4_survival_prognostics
import phase5_advisory_deployment


def run_phase0():
    print("\n" + "="*70)
    print(" [PHASE 0] SCOPE & FMECA DEFINITION")
    print("="*70)
    phase0_scope.main()


def run_phase1():
    print("\n" + "="*70)
    print(" [PHASE 1] DATA FOUNDATION, QUALITY AUDIT & PHYSICS WEAR NORMALIZATION")
    print("="*70)
    phase1_foundation.main()


def run_phase2():
    print("\n" + "="*70)
    print(" [PHASE 2] STATE DETECTION & NOVELTY WATCHLIST (NO WORK ORDERS NEEDED)")
    print("="*70)
    phase2_state_detection.main()


def run_phase3():
    print("\n" + "="*70)
    print(" [PHASE 3] SUPERVISED HEALTH ASSESSMENT (HORIZON SWEEP & ML CALIBRATION)")
    print("="*70)
    phase3_supervised_model.main()


def run_phase4():
    print("\n" + "="*70)
    print(" [PHASE 4] PROGNOSTICS (RECURRENT SURVIVAL & MULTI-HORIZON RISK)")
    print("="*70)
    phase4_survival_prognostics.main()


def run_phase5():
    print("\n" + "="*70)
    print(" [PHASE 5] ADVISORY GENERATION & COST-WEIGHTED DEPLOYMENT")
    print("="*70)
    phase5_advisory_deployment.main()


def main():
    parser = argparse.ArgumentParser(description="Oil Analysis Predictive Maintenance Master Pipeline Runner")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "0", "1", "2", "3", "4", "5"],
                        help="Phase to run (0, 1, 2, 3, 4, 5, or all)")
    args = parser.parse_args()

    phase_map = {
        "0": [run_phase0],
        "1": [run_phase1],
        "2": [run_phase2],
        "3": [run_phase3],
        "4": [run_phase4],
        "5": [run_phase5],
        "all": [run_phase0, run_phase1, run_phase2, run_phase3, run_phase4, run_phase5]
    }

    steps = phase_map[args.phase]
    print("Starting Predictive Maintenance Pipeline...")
    for step in steps:
        step()

    print("\n" + "="*70)
    print(" PIPELINE EXECUTION COMPLETE")
    print(" All artifacts generated in 'data/processed/' and 'artifacts/'")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
