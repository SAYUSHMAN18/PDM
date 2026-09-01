# Validation Report

## Scope

The uploaded project was tested against `data/current/SosFluidSample.xlsx` and `data/current/TelematicDataSample.xlsx`. The bundled `WorkOrderSample.csv` was treated as an example and excluded from the default real-data path.

## Corrected methodological defects

- Removed severity-derived and forced training labels.
- Removed the weighted rule/ML/anomaly score presented as probability.
- Removed classifier-derived remaining-safe-days output.
- Separated action priority from date-quality warnings.
- Required explicit WO outcomes, both target classes, minimum sample counts and matched assets before prediction.
- Added chronological train/calibration/test probability calibration.
- Excluded right-censored samples whose full future prediction horizon is not observable.
- Made interpretation categorisation multi-label with exact supporting sentences.
- Made external LLM processing explicit opt-in and identifier-free.
- Escaped spreadsheet-derived HTML values.
- Removed automatic loading of the example WO file.
- Fixed UTF-8 report serialization and current Streamlit API warnings.

## Version 3 usability and engineering improvements

- Removed the stale sidebar mode that could contradict the current analysis.
- Made the requested prediction horizon visible at all times and explicitly conditional on readiness gates.
- Replaced eight technical tabs with five task-based views.
- Collapsed repeated sample rows into one prioritised machine-component case.
- Added plain-language conclusions, required actions, timing, WO state and evidence confidence.
- Separated fleet action counts from raw S.O.S sample counts.
- Added per-machine laboratory trends, latest deltas, telemetry evidence and auditable raw records.
- Added a plain-language chronological holdout explanation and confusion matrix.
- Added issue → impact → required-fix data-quality reporting.
- Bundled an explicitly labelled matched synthetic demo for end-to-end prediction testing.

## Verified real-data result

| Check | Result |
|---|---:|
| Operating mode | Alert Management |
| S.O.S rows | 1,051 |
| Assets | 520 |
| Laboratory AR samples | 1,051 |
| HighPriority=T | 10 |
| New / Closed | 929 / 122 |
| WO linked / no link | 648 / 403 |
| New without WO | 364 |
| P1 Immediate / P1 Tracking | 3 / 6 |
| P2 Multiple Unlinked | 93 |
| P3 Action Required | 268 |
| Valid / invalid dates | 499 / 552 |
| Telemetry rows raw / cleaned | 952 / 259 |
| Matched assets | 0 |
| Failure probabilities generated | 0 |

## Commands executed

```text
PYTHONPATH=src python -m pytest -q
Result: 17 passed

python -m compileall -q app.py src tests
Result: passed with no errors

python -m predictive_maintenance.cli analyze --sos data/current/SosFluidSample.xlsx --telemetry data/current/TelematicDataSample.xlsx --output <temporary-output>
Result: Alert Management; UTF-8 output files written successfully

Streamlit AppTest
Result: all five pages passed with no application exceptions in both supplied-data and matched-demo modes (10 page/mode combinations)
```

## Remaining data limitations

- All S.O.S records are already AR; there is no normal comparison population.
- No structured numerical laboratory analytes are present.
- The telemetry asset does not match any S.O.S asset.
- WorkOrderId linkage is not a detailed or confirmed maintenance outcome.
- 552 sample dates have no usable calendar date.

The current deliverable is therefore an S.O.S alert-management and maintenance-triage application, not a validated failure-prediction model.
