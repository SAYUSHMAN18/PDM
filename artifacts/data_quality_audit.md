# Data Quality & Rule-Based Audit Report

Executive Summary:
The supplied S.O.S and Telematics extracts are sample datasets that **cannot be joined into a true predictive maintenance ML model** in their current form.

## 1. Equipment & Dataset Mismatch Matrix

| Feature | S·O·S Extract | Telematics Extract | Match Status |
|---|---|---|---|
| **Equipment IDs** | 120-000053, 120-000378 | 100-000064 | **ZERO MATCH** |
| **Serial Numbers** | 2ZR00294, BNH00655 | 63W02218 | **ZERO MATCH** |
| **Equipment Models** | 988F_CAT, 988G_CAT | 773B, 773 | **MISMATCHED** |
| **Work Orders** | Blank (`WorkOrderId` = NaN) | N/A | **MISSING** |
| **Numerical Lab Analytes** | Missing (Text Only) | N/A | **MISSING** |

> **Key Conclusion:** S.O.S samples belong to two Caterpillar wheel loaders (988F and 988G), while Telematics data belongs to a different Caterpillar 773B off-highway truck. No cross-joining or failure prediction is mathematically possible between these extracts.

## 2. Rule-Based S·O·S Diagnostic Audit (Wheel Loaders 120-000053 & 120-000378)

| Asset ID | Model | Compartment | Date Sampled | Meter Hours | Severity | Lab Diagnostics / Action Required |
|---|---|---|---|---|---|---|
| `120-000053` | `988F_CAT` | `FD_FR_LT` | 2025-12-30 | 12655 h | **AR** | Iron (Fe) Highly Elevated | Re-sample in 125h | Change Oil & Inspect Filters |
| `120-000053` | `988F_CAT` | `DIFF_FR` | 2025-12-30 | 12655 h | **AR** | Iron (Fe) Highly Elevated | Re-sample in 125h | Change Oil & Inspect Filters |
| `120-000053` | `988F_CAT` | `FD_FR_RT` | 2025-12-30 | 12655 h | **AR** | Iron (Fe) Highly Elevated | Re-sample in 125h | Change Oil & Inspect Filters |
| `120-000378` | `988G_CAT` | `WH_FR_RT` | 2025-12-15 | 25186 h | **AR** | Iron (Fe) Highly Elevated | Re-sample in 125h | Change Oil & Inspect Filters |
| `120-000378` | `988G_CAT` | `DIFF_RR` | 2025-12-15 | 25186 h | **AR** | Iron (Fe) Highly Elevated | Dirt Entry (Si + Al) | Re-sample in 125h | Change Oil & Inspect Filters |
| `120-000378` | `988G_CAT` | `WH_RR_RT` | 2025-12-15 | 25186 h | **AR** | Iron (Fe) Highly Elevated | Dirt Entry (Si + Al) | Re-sample in 125h | Change Oil & Inspect Filters |
| `120-000378` | `988G_CAT` | `WH_FR_LT` | 2025-12-15 | 25186 h | **AR** | Iron (Fe) Highly Elevated | Re-sample in 125h | Change Oil & Inspect Filters |
| `120-000378` | `988G_CAT` | `WH_RR_LT` | 2025-12-15 | 25186 h | **AR** | Iron (Fe) Highly Elevated | Dirt Entry (Si + Al) | Re-sample in 125h | Change Oil & Inspect Filters |

### Key Rule-Based Takeaways from S·O·S Data:
1. **100% Action Required Rate:** All 8 fluid samples carry an `AR` (Action Required) flag; zero healthy baseline samples exist for comparative training.
2. **Elevated Iron (Fe):** Present in all 8 compartments (Front/Rear Differentials, Final Drives, Wheel Ends).
3. **Dirt Ingress Alert:** Compartments `DIFF_RR`, `WH_RR_RT`, and `WH_RR_LT` on Asset `120-000378` exhibit combined Silicon (Si) and Aluminum (Al) elevation indicating seal failure or air induction leaks.
4. **Re-sampling Action:** Asset `120-000053` requires immediate oil change and re-sampling within **125 operating hours** to verify wear debris accumulation rates.

## 3. Telematics Audit (Cat 773B Truck 100-000064)

- **Total Telematics Records:** 952 rows spanning **2024-01-11 18:27:16 to 2026-08-25 23:08:25**.
- **Unique Timestamps:** 259 unique dates (duplicate periodic polls).
- **Operating Hours Logged:** 176.7 h to 2049.9 h (Delta: 1873.2 h).
- **Distance Travelled:** 257.7 km to 3873.3 km (Delta: 3615.6 km).
- **Usable Telemetry Fields:** `Location_Datetime, CumulativeOperatingHours_Hour, Distance_Odometer, Location_Latitude, Location_Longitude`.
- **Unusable / Stale Fields:** `FuelUsed, CumulativeIdleHours, EngineStatus (Stale/Constant), DEFRemaining, PayloadTotals` (constant values or unpopulated fields).

## 4. Required Datasets for Predictive ML Training

To train a honest 30-day predictive maintenance model (`failure_within_30_days`), request the following 4 matching datasets for the **SAME fleet and date range**:
1. **Raw S·O·S Lab Results:** Numerical PPM values for Fe, Cu, Al, Cr, Pb, Si, Na, K, Water, Soot, Fuel, Glycol, Viscosity, Oxidation, Nitration, TBN/TAN, lab severity across normal and abnormal samples.
2. **Work Orders (CMMS):** Work Order ID, Asset ID, Component, Open/Close Dates, PM/CM classification, Failure codes, Repair descriptions, Costs, Downtime.
3. **Matching Telematics:** Hour meters, idle hours, utilization rates for the **same Equipment IDs** as the S.O.S file.
4. **Asset Master Table:** Mapping between Asset ID, Serial Number, Equipment Number, Model, and Component naming conventions.