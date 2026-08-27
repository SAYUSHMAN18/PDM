"""
Phase 0 - Scope & FMECA  (weeks 1-3)

Decide what the pilot predicts, for which compartments, on which fleet -- before
any modelling.

  * Asset crosswalk        EquipNum <-> serial <-> model <-> site, from the asset
                           master (or derived from the S.O.S extract).
  * Compartment taxonomy   compartment code -> component family / position / sump.
  * FMECA shortlist        (component family x failure mode) pairs where oil
                           analysis genuinely carries the signal, ranked by
                           severity x cost. This is the model scope.

Outputs
  data/processed/asset_crosswalk.csv
  data/processed/compartment_taxonomy.json
  artifacts/phase0_scope_fmeca.json
"""

from __future__ import annotations

import json

import pandas as pd

from . import config
from .data import Extracts, load_extracts


def build_asset_crosswalk(ext: Extracts) -> pd.DataFrame:
    cols = ["machine_id", "serial", "model", "model_family", "site"]
    am = ext.assets.copy()
    for c in cols:
        if c not in am.columns:
            am[c] = "UNKNOWN"
    cw = am[cols].drop_duplicates("machine_id").reset_index(drop=True)
    cw = cw.rename(columns={"machine_id": "canonical_asset_id"})
    cw["equip_num"] = cw["canonical_asset_id"]

    counts = (ext.sos.groupby("machine_id")
              .agg(sos_samples=("sample_id", "size"),
                   compartments=("component", "nunique"),
                   first_sample=("sample_date", "min"),
                   last_sample=("sample_date", "max")))
    cw = cw.merge(counts, left_on="canonical_asset_id", right_index=True, how="left")
    cw["in_pilot"] = cw["sos_samples"].fillna(0) > 0
    return cw


def fmeca_table() -> pd.DataFrame:
    df = pd.DataFrame(config.FMECA_SHORTLIST)
    df["criticality_x_cost"] = df["typical_failure_cost_usd"] / df["severity_rank"]
    return df.sort_values("severity_rank").reset_index(drop=True)


def main(ext: Extracts | None = None) -> dict:
    config.ensure_dirs()
    ext = ext or load_extracts(verbose=False)

    crosswalk = build_asset_crosswalk(ext)
    crosswalk.to_csv(config.DATA_PROCESSED / "asset_crosswalk.csv", index=False)

    with open(config.DATA_PROCESSED / "compartment_taxonomy.json", "w") as f:
        json.dump(config.COMPARTMENT_TAXONOMY, f, indent=2)

    fmeca = fmeca_table()
    families_in_data = sorted(ext.sos["component_family"].unique())
    scoped = [m for m in config.FMECA_SHORTLIST if m["component_family"] in families_in_data]

    scope_doc = {
        "phase": "0 - Scope & FMECA",
        "pilot_fleet": {
            "machines": int(crosswalk["in_pilot"].sum()),
            "model_families": sorted(ext.sos["model_family"].unique()),
            "sites": sorted(ext.sos["site"].unique()),
            "sampled_compartment_families": families_in_data,
        },
        "fmeca_shortlist": json.loads(fmeca.to_json(orient="records")),
        "in_scope_failure_modes": len(scoped),
        "out_of_scope_note": config.OUT_OF_SCOPE_NOTE,
        "primary_prediction_target": (
            f"Corrective work order on the same machine + compartment within "
            f"{config.PRIMARY_HORIZON} days of an S.O.S sample."),
        "gate": "PASS" if scoped else "BLOCKED - no in-scope families present in the extract",
    }
    with open(config.ARTIFACTS / "phase0_scope_fmeca.json", "w") as f:
        json.dump(scope_doc, f, indent=2, default=str)

    print("Phase 0 - Scope & FMECA")
    print(f"  pilot machines        : {scope_doc['pilot_fleet']['machines']}")
    print(f"  model families        : {', '.join(scope_doc['pilot_fleet']['model_families'])}")
    print(f"  compartment families  : {', '.join(families_in_data)}")
    print(f"  FMECA in-scope modes  : {len(scoped)} / {len(config.FMECA_SHORTLIST)}")
    print(f"  prediction target     : corrective WO within {config.PRIMARY_HORIZON} days")
    print(f"  gate                  : {scope_doc['gate']}")
    return scope_doc


if __name__ == "__main__":
    main()
