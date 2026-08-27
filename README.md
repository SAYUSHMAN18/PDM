# S·O·S + Work Order → 30-day maintenance risk model

A complete, runnable starter for heavy-machinery predictive maintenance built on
oil analysis and maintenance history. It runs today on synthetic data; when your
real extracts arrive you delete one script and change nothing else.

## The idea in one paragraph

An oil sample is a blood test: it says what metal is coming off which part, what
dirt or water got in, and whether the oil still protects. A work order is the
medical record: what actually got repaired, and when. Line them up in time and
you get thousands of little experiments — *"the oil looked like this, and 22 days
later the engine came in for a bearing job."* The model learns those patterns and
then, for each new sample, answers one question: **how likely is a corrective
repair on this component in the next 30 days?**

## Run Pipelines & Data Audit

### 1. Data Quality & Rule-Based Audit (Current Sample Extracts)
Run rule-based diagnostics, detect elevated Iron / Dirt entry, and evaluate sample suitability:
```bash
python audit_sample_data.py       # outputs artifacts/data_quality_audit.md
```

### 2. S·O·S Oil Analysis & Work Order Pipeline (30-Day Risk ML Model)
```bash
python build_dataset.py          # join, label, engineer features from raw CSVs
python train_model.py            # time-split training + model selection
python score_new_samples.py      # risk cards for every machine-component
```

### 3. Enterprise Data Request Specification
Before ML training on live enterprise data, request matching S·O·S, Telematics, Work Order, and Asset Master exports following [`DATA_REQUEST_ENTERPRISE.md`](file:///c:/Users/ersay/Downloads/sos_wo_pdm_starter/DATA_REQUEST_ENTERPRISE.md).

## What each file does

| File | Role |
|---|---|
| `audit_sample_data.py` | Rule-based diagnostic report & data quality audit on enterprise sample files |
| `DATA_REQUEST_ENTERPRISE.md` | Formal 4-dataset request specification & message template for data providers |
| `build_dataset.py` | S.O.S + Work Order labelling, leakage guards, feature engineering |
| `train_model.py` | 30-day corrective maintenance risk model training & evaluation |
| `score_new_samples.py` | Scores latest oil samples per machine-component, prints risk cards |
| `build_sensor_dataset.py` | Extracts cycle summary features from 17 sensor `.txt` files |
| `train_sensor_model.py` | Trains component condition models (Cooler, Valve, Pump, Accumulator) |
| `make_synthetic_data.py` | Synthetic S.O.S + WO generator (optional benchmark) |

## How the label is made

For each sample, look forward 30 days. Was a **corrective** (`CM`) work order
opened on the *same machine and same component*? Yes → 1, no → 0.

Three guards stop the model from cheating or learning nonsense:

- **Preventive work orders are never positives.** A scheduled oil change is maintenance, not failure.
- **Censoring**: samples in the last 30 days of the data have no future to look into — dropped.
- **Post-repair blackout**: samples within 7 days after a repair are dropped; the oil is disturbed and the outcome is already known.

## Where the predictive power actually comes from

Not the raw numbers. Iron at 90 ppm means nothing on its own — it depends on the
component, the oil age and the machine. What the model uses:

- **Trend**: change since last sample, % change, 3-sample slope
- **Wear rate**: ppm added per 100 machine hours
- **Own baseline**: value ÷ this machine-component's rolling median (of *previous* samples only)
- **Peer baseline**: robust z-score against the same model + component across the fleet
- **Context**: oil hours, service meter, days since last repair, corrective repairs in the last year, the lab's own severity flag

Ratios matter too: silicon with iron says dirt ingress; copper with lead says
bearings; sodium and potassium with water says coolant.

On the synthetic fleet, permutation importance ranks `pq_index__peer_z`,
`fe_ppm__peer_z`, `cu_ppm__delta` and `soot_pct__vs_own_med` at the top — i.e.
deviation and movement, not absolute level. Expect the same shape on real data.

## How it is evaluated (and why accuracy is the wrong metric)

Roughly 3% of samples are followed by a repair. A model that predicts "no failure"
for everything is 97% accurate and completely useless. So:

- **PR-AUC** against the 3% base rate (the lift number is what to quote)
- **Precision / recall at top-10%** — you can only inspect so many machines a week
- **The lab's severity flag as a baseline.** If the model can't beat "ACTION or CRITICAL", ship the flag and save the effort.
- **Time-based split**: train on the earlier window, test on the later one. A random split leaks the future into the past and produces beautiful, fake scores.

Synthetic-data result: lab flag PR-AUC 0.37 → logistic regression 0.67 (≈23× the
base rate), catching ~88% of failures inside the top 10% of samples. Real data
will be messier; treat these numbers as a template for the report, not a target.

## What this model is not

It is an **early-warning and prioritisation tool**, not an automatic decision-maker,
and not a remaining-useful-life estimate. Known limits, all real:

- Work orders are raised days after the actual problem — the horizon is fuzzy at the edges.
- One work order can cover several components.
- A repair done before sampling changes the oil result.
- Different machines and duty cycles have genuinely different normal levels.
- Samples taken *because* someone was worried are not random samples.

Ship it as a ranked weekly list for the reliability engineer to review, log whether
each alert was right, and use that log as your next training set.

## Roadmap once the 30-day model works

1. Calibrate probabilities (`CalibratedClassifierCV`) so "78%" means something.
2. Predict the **failure code**, not just yes/no — turns an alert into a work instruction.
3. Attach cost: `risk × (downtime + parts + labour)` ranks by money, which is what gets budget approved.
4. Add machine telemetry (load, temperature, pressure) between oil samples.
5. Only then attempt remaining useful life.
