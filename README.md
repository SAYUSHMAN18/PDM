# Oil Analysis Predictive Maintenance Pilot — Implementation Roadmap

A complete 6-month implementation roadmap and software foundation for heavy-machinery predictive maintenance built on oil analysis, telematics, and maintenance history.

## Architectural Strategy: Value Before Work Orders

The implementation is structured into **6 progressive phases**. **Phases 0–2 require zero work order data** and deliver immediate engineering value (ASTM D7720 alarms, EWMA/CUSUM change detection, multivariate novelty scores, `InterpText` mining, and weekly ranked watchlists) before CMMS work order extracts land.

```text
Phase 0: Scope & FMECA (Weeks 1–3) -> python phase0_scope.py
      ↓
Phase 1: Data Foundation & Quality Audit (Weeks 3–7) -> python phase1_foundation.py
      ↓
Phase 2: State Detection — First Thing That Ships (Weeks 6–11, NO Work Orders Needed) -> python phase2_state_detection.py
      ↓
Phase 3: Health Assessment — Supervised ML Model (Weeks 10–18, Needs Work Orders) -> python phase3_supervised_model.py
      ↓
Phase 4: Prognostics — Survival with Frailty (Weeks 16–22) -> python phase4_survival_prognostics.py
      ↓
Phase 5: Advisory Generation & Deployment (Weeks 20–27) -> python phase5_advisory_deployment.py
      ↓
Phase 6: Advanced RUL & Telematics Integration (Week 28+)
```

---

## Single Unified Execution Command

Run the entire 6-phase pipeline end-to-end with one command:
```bash
python run_pipeline.py
```

To run a specific phase individually:
```bash
python run_pipeline.py --phase 2   # Runs Phase 2 (State Detection & Watchlist)
```

---

## Modular File Structure & Roles

| File | Phase | Role |
|---|---|---|
| [`run_pipeline.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/run_pipeline.py) | Master | Master entry-point runner for executing all 6 pipeline phases |
| [`phase0_scope.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/phase0_scope.py) | Phase 0 | Pilot fleet FMECA, asset crosswalk, and compartment taxonomy decoder |
| [`phase1_foundation.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/phase1_foundation.py) | Phase 1 | Data quality audit, label governance audit, and wear mass rate normalization |
| [`phase2_state_detection.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/phase2_state_detection.py) | Phase 2 | ASTM D7720 limits, EWMA change detection, novelty scores, and weekly watchlist |
| [`phase3_supervised_model.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/phase3_supervised_model.py) | Phase 3 | Horizon sweep (14/30/60/90d), leakage guards, model training & calibration |
| [`phase4_survival_prognostics.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/phase4_survival_prognostics.py) | Phase 4 | Recurrent-event survival frailty prognostics & multi-horizon risk scoring |
| [`phase5_advisory_deployment.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/phase5_advisory_deployment.py) | Phase 5 | Cost-weighted expected risk ranking ($\text{Risk} \times \text{Cost}$) & ROI valuation |
| [`audit_sample_data.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/audit_sample_data.py) | Phase 0–1 | S.O.S & Telematics enterprise sample audit & NLP text mining |
| [`DATA_REQUEST_ENTERPRISE.md`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/DATA_REQUEST_ENTERPRISE.md) | Phase 0 | Formal 4-dataset request specification & email template for data owners |
| [`build_sensor_dataset.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/build_sensor_dataset.py) | Phase 6 | High-frequency physical sensor feature extraction (17-sensor rig) |
| [`train_sensor_model.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/train_sensor_model.py) | Phase 6 | Hardware rig component condition classifiers (Cooler, Valve, Pump, Accumulator) |

---

## Detailed Implementation Plan

The complete 6-month implementation roadmap document is available in [`implementation_plan.md`](file:///C:/Users/ersay/.gemini/antigravity-ide/brain/52ec7b2d-a7fc-492a-ba0d-774527ba06b0/implementation_plan.md).
