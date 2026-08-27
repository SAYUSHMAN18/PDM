# Oil Analysis Predictive Maintenance Pilot — Implementation Roadmap

A complete 6-month implementation roadmap and software foundation for heavy-machinery predictive maintenance built on oil analysis, telematics, and maintenance history.

## Architectural Strategy: Value Before Work Orders

The implementation is structured into **6 progressive phases**. **Phases 0–2 require zero work order data** and deliver immediate engineering value (ASTM D7720 alarms, EWMA/CUSUM change detection, multivariate novelty scores, `InterpText` mining, and weekly ranked watchlists) before CMMS work order extracts land.

```text
Phase 0: Scope & FMECA (Weeks 1–3)
      ↓
Phase 1: Data Foundation & Quality Audit (Weeks 3–7)
      ↓
Phase 2: State Detection — First Thing That Ships (Weeks 6–11, NO Work Orders Needed)
      ↓
Phase 3: Health Assessment — Supervised ML Model (Weeks 10–18, Needs Work Orders)
      ↓
Phase 4: Prognostics — Survival with Frailty (Weeks 16–22)
      ↓
Phase 5: Advisory Generation & Deployment (Weeks 20–27)
      ↓
Phase 6: Advanced RUL & Telematics Integration (Week 28+)
```

---

## Quick Start & Code Structure

### 1. Data Audit & Rule-Based Diagnostics (Phases 0–1)
Run data quality profiling, NLP interpretation text mining, and cross-file alignment:
```bash
python audit_sample_data.py       # outputs artifacts/data_quality_audit.md
```

### 2. State Detection & Novelty Scoring (Phase 2 — No Work Orders Required)
Calculates ASTM D7720 baseline limits, EWMA change detection, and multivariate novelty scores:
```bash
python build_dataset.py          # cleans, normalizes wear rates, and builds feature table
```

### 3. Supervised Model & Live Scoring (Phase 3 — Requires Work Orders)
```bash
python train_model.py            # time-split training & model evaluation
python score_new_samples.py      # risk cards & cost-ranked watchlist
```

---

## Key Files & Roles

| File | Phase | Role |
|---|---|---|
| [`audit_sample_data.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/audit_sample_data.py) | Phase 0–1 | Data quality audit, NLP text mining, and cross-file alignment report |
| [`DATA_REQUEST_ENTERPRISE.md`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/DATA_REQUEST_ENTERPRISE.md) | Phase 0 | Formal 4-dataset request specification & provider email template |
| [`build_dataset.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/build_dataset.py) | Phase 1–2 | Mass-balance wear normalization, rolling trend, own/peer Z-scores |
| [`train_model.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/train_model.py) | Phase 3 | Time-split supervised model training, thresholding & calibration |
| [`score_new_samples.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/score_new_samples.py) | Phase 3–5 | Cost-ranked weekly watchlist cards & risk probability scoring |
| [`build_sensor_dataset.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/build_sensor_dataset.py) | Phase 6 | High-frequency physical sensor feature extraction (17-sensor rig) |
| [`train_sensor_model.py`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/train_sensor_model.py) | Phase 6 | Component condition classification (Cooler, Valve, Pump, Accumulator) |

---

## Detailed Implementation Plan

The complete 6-month roadmap document is available in [`implementation_plan.md`](file:///C:/Users/ersay/.gemini/antigravity-ide/brain/52ec7b2d-a7fc-492a-ba0d-774527ba06b0/implementation_plan.md).
