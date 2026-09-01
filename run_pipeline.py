"""
Master runner for the oil-analysis predictive-maintenance pilot.

    python run_pipeline.py                 # phases 0-5 end to end
    python run_pipeline.py --phase 2       # one phase
    python run_pipeline.py --synth         # (re)generate synthetic data, then run all
    python run_pipeline.py --synth --seed 7

Phases 0-2 need no work orders and ship real value (D7720 alarms, change
detection, novelty scores, a weekly watchlist). Phases 3-5 need the work-order
extract; they skip themselves cleanly when it is absent.

Drop real extracts into data/raw/ as sos_samples.csv / work_orders.csv /
asset_master.csv (contract in DATA_REQUEST.md) and delete data/raw/.synthetic --
the pipeline then runs on the real data with no code change.
"""

from __future__ import annotations

import argparse
import time

from pdm import (config, phase0_scope, phase1_foundation, phase2_state_detection,
                 phase3_supervised, phase4_survival, phase5_advisory)
from pdm.data import load_extracts

PHASES = {
    "0": ("Scope & FMECA", phase0_scope.main),
    "1": ("Data foundation & quality audit", phase1_foundation.main),
    "2": ("State detection & weekly watchlist (no work orders)", phase2_state_detection.main),
    "3": ("Supervised health assessment", phase3_supervised.main),
    "4": ("Survival prognostics (30/60/90-day risk)", phase4_survival.main),
    "5": ("Advisory generation & cost-weighted deployment", phase5_advisory.main),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all", choices=["all", *PHASES])
    ap.add_argument("--synth", action="store_true",
                    help="regenerate synthetic data/raw/*.csv before running")
    ap.add_argument("--seed", type=int, default=None, help="synthetic-data seed")
    args = ap.parse_args()

    config.ensure_dirs()

    if args.synth:
        from pdm import synth
        print("Generating synthetic data ...")
        out = synth.write(args.seed) if args.seed is not None else synth.write()
        cm = int((out["work_orders"]["WOType"] == "CM").sum())
        print(f"  sos_samples.csv : {len(out['sos']):,} rows")
        print(f"  work_orders.csv : {len(out['work_orders']):,} rows ({cm:,} corrective)")
        print(f"  asset_master.csv: {len(out['assets']):,} machines\n")

    ext = load_extracts(verbose=True)          # load + canonicalise once, share across phases
    print()

    steps = list(PHASES) if args.phase == "all" else [args.phase]
    t0 = time.time()
    for key in steps:
        title, fn = PHASES[key]
        print("=" * 72)
        print(f" PHASE {key} - {title}")
        print("=" * 72)
        fn(ext)
        print()

    print("=" * 72)
    print(f" PIPELINE COMPLETE in {time.time() - t0:.0f}s   "
          f"(artifacts/ and data/processed/)")
    print("=" * 72)


if __name__ == "__main__":
    main()
