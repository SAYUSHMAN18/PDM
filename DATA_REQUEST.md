# What to ask for when you request the S·O·S and work-order extracts

Hand this page to whoever owns the oil-lab portal and the CMMS. The column names
are the contract the code expects — if the source uses different names, rename on
export or map them in `load_raw()`.

## File 1 — `sos_samples.csv` (one row per oil sample)

| Column | Type | Why the model needs it |
|---|---|---|
| `sample_id` | text | de-duplicate re-issued lab reports |
| `machine_id` | text | join key to work orders — **must match the CMMS asset ID exactly** |
| `machine_type`, `model` | text | peer comparison (an iron level that is normal for a haul truck is not normal for an excavator) |
| `component` | text | ENGINE / HYDRAULIC / TRANSMISSION / FINAL_DRIVE — the model is per component |
| `sample_date` | date | the clock everything is aligned to |
| `smu_hours` | number | machine service meter at sampling |
| `oil_hours` | number | hours on the *oil* since the last change — without this, iron is uninterpretable |
| `oil_changed` | Y/N | flags the reset point |
| `fe_ppm cu_ppm cr_ppm pb_ppm al_ppm` | number | wear metals — which part is wearing |
| `si_ppm na_ppm k_ppm` | number | contamination — dirt, coolant |
| `water_pct fuel_pct glycol_pct soot_pct` | number | contamination and combustion |
| `visc40 oxidation nitration tbn` | number | is the oil still doing its job |
| `pq_index` | number | large ferrous debris (catches what ICP misses) |
| `lab_severity` | NORMAL/MONITOR/ACTION/CRITICAL | the lab's own call — your benchmark to beat |

Ask for **at least 24–36 months** of history and *all* samples, not just flagged ones.
A model trained only on flagged samples never learns what normal looks like.

## File 2 — `work_orders.csv` (one row per work order)

| Column | Type | Why |
|---|---|---|
| `wo_id` | text | key |
| `machine_id` | text | join key |
| `component` | text | same vocabulary as the sample file — this is the single most common data problem |
| `wo_type` | PM / CM | **critical**: a scheduled oil change is not a failure |
| `failure_code` | text | later: predict *which* failure, not just whether |
| `description` | text | sanity-check the labels by eye |
| `open_date`, `close_date` | date | open_date defines the event; the gap is downtime |
| `smu_at_wo` | number | cross-check against sample hours |
| `downtime_hours`, `parts_cost`, `labour_cost` | number | lets you price a prevented failure and justify the project |

## Five questions to ask the data owner before you model anything

1. **Is `wo_type` reliable?** If PM/CM is mislabelled or blank, every label is wrong. Ask for the raw maintenance-type field too.
2. **How late are work orders raised?** A 3-day reporting lag is normal; a 3-week lag means your 30-day horizon is really a 10-day horizon.
3. **Does one work order cover several components?** If so, ask for the WO task/line-item table, not just the header.
4. **Do machine IDs survive a component swap?** If an engine is transplanted between machines, its wear history moves with it — you need the component serial, not just the machine ID.
5. **Are there samples taken *because* someone suspected a problem?** Those are gold for labels but bias the model — flag them if the field exists (`sample_reason`).
