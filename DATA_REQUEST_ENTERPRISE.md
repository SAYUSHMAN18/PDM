# Enterprise Data Request Specification for Predictive Maintenance

**To:** Data Governance / Enterprise CMMS & Telematics Team  
**Subject:** Data Extract Request for 30-Day Predictive Maintenance Model (S·O·S, Telematics, Work Orders & Asset Master)

---

## Executive Summary & Current Extract Findings

An audit of the initial sample files (`SosFluidSample.xlsx` and `TelematicDataSample.xlsx`) revealed that the sample extracts contain mismatched assets (S·O·S wheel loaders `120-000053` / `120-000378` vs. Telematics 773B truck `100-000064`), missing numerical laboratory measurement values (text interpretations only), unlinked work order IDs, and zero normal baseline samples.

To build an accurate **30-day Machine-Component Predictive Maintenance ML Model**, we require 4 matching export files covering the **same equipment fleet and historical time window**.

---

## The 4 Required Datasets

```text
Telemetry History ──┐
                    ├──> Asset ID + Component + Date ──> 30-Day Failure Prediction (Label: 0/1)
Raw S·O·S Lab Data ──┤
                    │
Work Orders (CMMS) ─┘
```

### 1. Raw S·O·S Fluid Analysis Export (`sos_samples_raw.csv` / `.xlsx`)
For every oil sample taken across the target fleet:
* **Canonical Asset ID & Serial Number** (e.g., `120-000053`, `2ZR00294`)
* **Component / Compartment** (`ENGINE`, `TRANSMISSION`, `HYDRAULIC`, `DIFF_FR`, `FD_FR_LT`, etc.)
* **Sample Date & Processed Date**
* **Equipment Meter Hours (SMU)** & **Fluid Hours**
* **Wear Metals (PPM):** Iron (`Fe`), Copper (`Cu`), Aluminum (`Al`), Chromium (`Cr`), Lead (`Pb`), Silicon (`Si`), Sodium (`Na`), Potassium (`K`)
* **Contaminants & Physicals:** Water (`%`), Soot (`%`), Fuel Dilution (`%`), Glycol (`%`), Viscosity @ 40°C, Oxidation, Nitration, TBN / TAN, PQ Index (Ferrous Debris)
* **Fluid Changed Indicator** (`Y`/`N`) & **Filter Changed Indicator** (`Y`/`N`)
* **Laboratory Severity Flag** (`NORMAL`, `MONITOR`, `ACTION`, `CRITICAL`) & **Chemist Interpretation Text**

*Note: Please include historical samples spanning both normal operation and abnormal wear episodes for each machine.*

### 2. Work Orders Export from CMMS (`work_orders.csv` / `.xlsx`)
All maintenance work orders raised for the fleet over the same timeframe:
* **Work Order ID**
* **Canonical Asset ID & Serial Number**
* **Component / Compartment**
* **Work Order Open Date & Close Date**
* **Work Order Classification:** `CM` (Corrective Maintenance) vs `PM` (Preventive Maintenance)
* **Failure / Problem Codes** & **Technician Description**
* **Action Taken & Parts Repaired / Replaced**
* **Meter Hours at Repair (SMU)**
* **Downtime Hours**, **Parts Cost**, and **Labour Cost**

### 3. Matching Telematics Export (`telematics_daily.csv` / `.xlsx`)
Telematics logs for the **same Equipment IDs** as the S·O·S file:
* **Canonical Asset ID & Serial Number**
* **Timestamp** (Date / Time)
* **Cumulative Operating Hours**
* **Cumulative Idle Hours** & **Engine Status**
* **Distance Travelled / Odometer**
* **Fuel Consumed** & **Daily Utilization**

### 4. Asset Master Mapping Table (`asset_master.csv` / `.xlsx`)
Unified mapping cross-referencing all system identifiers:
* **Canonical Asset ID** (e.g., `120-000053`)
* **Equipment Number** (e.g., `120-000053`)
* **Serial Number** (e.g., `2ZR00294`)
* **Equipment Model** (e.g., `988F`) & **Manufacturer** (`CATERPILLAR`)
* **Site / Jobsite Name** & **Active Date Ranges**

---

## Ready-to-Send Email / Message Template for Data Provider

```text
Dear Data Engineering Team,

We have completed the data quality audit on the initial sample extracts (SosFluidSample.xlsx and TelematicDataSample.xlsx). 

To proceed with building the 30-day Predictive Maintenance Machine Learning model, we need matching export files covering the SAME equipment fleet and historical date range. 

Currently, the sample S.O.S file covers wheel loaders (120-000053 and 120-000378) without numerical laboratory PPM values, while the telematics file covers a different machine (100-000064, 773B truck).

Please provide standard CSV or Excel exports for the following 4 datasets covering the same equipment:
  1. Raw S.O.S Results: Numerical PPM wear metals (Fe, Cu, Al, Si, Cr, Pb), contaminants (water, soot, fuel, glycol), viscosity, TBN/TAN, and lab severity flags across both normal and abnormal samples.
  2. Work Orders (CMMS): Work Order ID, Equipment ID, Component, Open/Close dates, PM/CM classification, Failure codes, Parts replaced, Downtime, and Repair costs.
  3. Matching Telematics: Operating hours, idle hours, and distance logs for the same equipment IDs contained in the S.O.S file.
  4. Asset Master Table: Mapping between Asset ID, Serial Number, Equipment ID, Model, and Site.

Thank you!
```
