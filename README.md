# Heavy Machinery Condition Monitoring

This project analyses S.O.S fluid records, optional telemetry and independently confirmed work-order outcomes in three guarded operating modes. Version 3 presents the result as a machine-level engineering decision workflow instead of a wall of sample-level alerts.

## Start here

On Windows or in the Antigravity terminal:

```powershell
cd "C:\Users\ersay\Downloads\Heavy_Machinery_Predictive_Maintenance"
.\setup_and_run.bat
```

Open `http://localhost:8501`, select a data source, and choose **Run analysis**. Use:

- **Supplied current data** to analyse the provided alert-only export.
- **Matched prediction demo** to test S.O.S + telemetry + detailed WO prediction end to end. A warning identifies it as synthetic.
- **Upload your own files** to analyse your exports without changing the source code.

The dashboard has five views: Fleet Overview, Action Queue, Asset Analysis, Model Validation and Data Quality.

## What the current data can do

`data/current/SosFluidSample.xlsx` is an alert-only export. All 1,051 rows have `OverallInterp=AR`, so the application runs in **Alert Management** mode and does not calculate failure probabilities.

Verified current-data results:

- 520 assets and 1,051 laboratory AR records
- 10 `HighPriority=T` records
- 929 New and 122 Closed laboratory records
- 648 records with a linked `WorkOrderId`; 403 without one
- 364 New records without a WO link
- 3 P1 Immediate Review records and 6 P1 WO Tracking records
- 93 P2 Multiple Unlinked records after higher-priority overrides
- 499 valid sample dates and 552 time-only/invalid dates
- 952 telemetry rows reduced to 259 snapshots, but no S.O.S/telemetry asset IDs match
- No structured Fe, Cu, Si, water, viscosity or other numerical laboratory columns

`WorkOrderId` linkage does not prove inspection, repair, failure confirmation or closure.

## Operating modes

1. **Alert Management** — prioritises laboratory records and maintenance follow-up. No ML probability.
2. **Condition Monitoring** — requires at least one asset/component series with three dated numerical laboratory samples.
3. **Failure Prediction** — additionally requires normal and abnormal history, at least 60 independently labelled samples, explicit confirmed WO outcomes with both classes, at least five matched telemetry assets covering at least 30 S.O.S rows, and successful chronological train/calibration/test validation.

The model never creates labels from S.O.S severity and never flips labels to force two classes.

## Run on Windows

Open PowerShell or the Antigravity terminal:

```powershell
cd "C:\Users\ersay\Downloads\Heavy_Machinery_Predictive_Maintenance"
.\setup_and_run.bat
```

The setup script creates `.venv`, installs dependencies, runs the tests and starts Streamlit. Open `http://localhost:8501` if the browser does not open automatically.

Later launches:

```powershell
.\run_dashboard.bat
```

Stop the server with `Ctrl+C`.

## Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH="$PWD\src"
python -m pytest -q
python -m streamlit run app.py
```

The deterministic maintenance summary is local by default. Optional Gemini support requires:

```powershell
python -m pip install -r requirements-ai.txt
```

External processing must then be explicitly enabled in the dashboard. Asset identifiers are excluded from the external prompt.

## Command-line analysis

```powershell
$env:PYTHONPATH="$PWD\src"
python -m predictive_maintenance.cli analyze `
  --sos data\current\SosFluidSample.xlsx `
  --telemetry data\current\TelematicDataSample.xlsx `
  --output outputs\current
```

Expected CLI mode: `Alert Management` with zero matched assets.

## Important outputs

- `analysis_summary.json` — data coverage and operational counts
- `model_readiness.csv` — explicit PASS/BLOCKED gates
- `sos_engineering_alerts.csv` — evidence and workflow priorities
- `ai_insights.md` — deterministic maintenance triage summary
- `telemetry_cleaned_scored.csv` — cleaned telemetry; anomaly scoring remains disabled unless prediction readiness passes

## Methodological protections

- `AR` is never treated as a predicted failure.
- Rule-match strength is never shown as a probability.
- Work-order labels come only from future confirmed corrective outcomes.
- Samples without a fully observable future horizon are excluded instead of being labelled negative.
- Provide `ObservationEndDate` (or `DataThroughDate`) in the detailed WO export when available; otherwise the latest explicit WO date is used conservatively as the observation boundary.
- A classifier never invents remaining safe days or remaining useful life.
- Model probability is not blended with rule or anomaly scores.
- Model evaluation uses chronological train, sigmoid-calibration and untouched test blocks.
- Invalid dates are flagged separately and never suppress P1 action priority.
- Closed laboratory status does not claim that maintenance was completed.
- External LLM processing is off by default.

## Validation

The included suite covers real-workbook metrics, strict date parsing, independent priority/data-quality logic, multi-label evidence extraction, label leakage prevention, readiness gates, chronological model calibration, UTF-8 CLI output and a Streamlit dashboard smoke test.

```powershell
python -m pytest -q
```

Expected result: `17 passed`.

Synthetic or example work orders are schema demonstrations only and must not be mixed into real fleet analysis.
