"""
Oil-analysis predictive maintenance pilot for heavy machinery.

Everything the pipeline needs lives in this package:

    config    - paths, taxonomies, cost model, FMECA scope
    synth     - realistic S.O.S + work-order + asset generator (enterprise schema)
    data      - load raw extracts (synthetic OR real) and canonicalise them
    features  - labelling (leakage guards) + feature engineering
    phase0..5 - the six pilot phases

Run the whole thing with ``python run_pipeline.py`` from the repo root.
"""

__version__ = "1.0.0"
