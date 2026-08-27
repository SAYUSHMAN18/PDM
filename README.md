# Oil-Analysis Predictive Maintenance Pilot

A working pipeline that turns **S·O·S fluid analysis** + **work orders** into a
weekly, cost-ranked maintenance watchlist for heavy machinery (CAT wheel loaders,
haul trucks, excavators). Built as six progressive phases that mirror the
ISO 13374 / OSA-CBM reference architecture.

The design rule: **Phases 0–2 need no work orders** and already ship value
(fleet-relative alarm limits, time-aware change detection, novelty scores, a
ranked watchlist). Phases 3–5 add supervised risk, survival prognostics and a
cost-weighted advisory once the CMMS extract lands — and skip themselves cleanly
until it does.

```
Phase 0  Scope & FMECA .......... which compartments, which failure modes, which fleet
Phase 1  Data foundation ........ canonical table, label-governance audit, ppm→mass physics
Phase 2  State detection ........ ASTM D7720 limits · time-aware EWMA · Mahalanobis novelty
                                  · InterpText mining · weekly watchlist   [no work orders]
Phase 3  Health assessment ..... horizon sweep · leakage guards · calibrated GBM vs rule
Phase 4  Prognostics ........... discrete-time survival hazard · 30/60/90-day risk · frailty
Phase 5  Advisory & deployment . cost-weighted ranking (risk × failure cost) · ROI · feedback
```

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py --synth        # generate demo data, then run all six phases
```

```bash
python run_pipeline.py               # re-run on whatever is in data/raw/
python run_pipeline.py --phase 2     # run one phase
python -m pdm.score --top 15         # risk cards for the latest sample per machine-compartment
```

## Running on real data

Drop `sos_samples.csv`, `work_orders.csv`, `asset_master.csv` into `data/raw/`
(column contract in [`DATA_REQUEST.md`](DATA_REQUEST.md)), delete
`data/raw/.synthetic`, and re-run `python run_pipeline.py`. No code change — the
loader auto-detects synthetic vs. real vs. sample-only mode and prints which one
it is using, plus warnings when chemistry or work orders are missing.

## Layout

| Path | Role |
|---|---|
| `pdm/config.py` | one place for paths, taxonomies, cost model, FMECA scope, horizons |
| `pdm/synth.py` | realistic synthetic S·O·S + WO + asset generator (enterprise schema) |
| `pdm/data.py` | load raw extracts (synthetic **or** real) → canonical internal schema |
| `pdm/features.py` | labelling with leakage guards + within-oil-run trend/physics features |
| `pdm/model.py` | shared time-split, evaluation protocol (PR-AUC, precision@capacity), thresholds |
| `pdm/phase0_scope.py` … `phase5_advisory.py` | the six phases |
| `pdm/score.py` | score new samples with the trained Phase 3 model |
| `run_pipeline.py` | master runner |
| `artifacts/`, `data/processed/` | generated outputs (git-ignored) |

## Outputs worth looking at

- `data/processed/phase2_watchlist.csv` — the weekly ranked list (ships first)
- `artifacts/d7720_population_limits.csv` — fleet limits vs. generic OEM condemning limits
- `artifacts/phase3_metrics.txt` — horizon sweep + model-vs-rule comparison on a time split
- `artifacts/phase4_multi_horizon_risk.csv` — 30/60/90-day survival risk per machine-compartment
- `data/processed/phase5_cost_ranked_advisories.csv` — the P1/P2/P3 worklist with expected value
- `artifacts/feedback_capture_table.csv`, `artifacts/advisory_feedback_table.csv` — alert-outcome
  logs; six months of these become a clean supervised label set

## Notes & honest limits

- **No `xgboost` / `catboost` / `lifelines` dependency.** Phase 3 uses
  scikit-learn `HistGradientBoostingClassifier` with monotonic constraints
  (rising iron can never lower predicted risk); Phase 4 is a pooled-logistic
  discrete-time hazard. Swapping in CatBoost / an Andersen-Gill frailty model is
  a Phase 6 upgrade, not a rewrite.
- The synthetic generator is for wiring and demos. Real fault dynamics, reporting
  lag and label noise are only approximated.
- Phase 4 frailty is a shrinkage proxy (trailing-year machine WO rate), not a
  fitted random effect.
- Treat the system as an **early-warning and prioritisation** tool, not an
  automatic decision-maker, until the Phase 1 label-governance gate passes on
  real data.
