from pathlib import Path
import numpy as np
import pandas as pd

from predictive_maintenance.data import load_table, prepare_sos, prepare_telemetry, prepare_work_orders
from predictive_maintenance.features import build_training_table
from predictive_maintenance.llm import generate_llm_insights
from predictive_maintenance.models import train_failure_models
from predictive_maintenance.pipeline import (
    MODE_ALERT, MODE_CONDITION, MODE_PREDICTION,
    P1_IMMEDIATE, P1_WO_TRACKING, P2_REPEATED, P3_ACTION,
    TIER_IN_PROGRESS, TIER_CLOSED,
    analyze_frames, run_analysis,
)
from predictive_maintenance.presentation import (
    build_case_table, build_fleet_summary, build_mode_summary,
    build_validation_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def test_sos_sample_automatically_selects_alert_management_mode():
    result = analyze_frames(
        load_table(ROOT / "data/current/SosFluidSample.xlsx"),
        load_table(ROOT / "data/current/TelematicDataSample.xlsx"),
    )
    assert result["operating_mode"] == MODE_ALERT
    assert result["mode"] == MODE_ALERT
    assert not result["predictive_risk_enabled"]
    assert len(result["sos"]) == 1051
    assert len(result["alerts"]) == 1051
    assert result["matched_assets"] == set()


def test_ar_alone_never_produces_failure_probability():
    ar_sos = pd.DataFrame({
        "SampleNum": ["S999"],
        "EquipNum": ["EQ-100"],
        "Compartment": ["ENGINE"],
        "OverallInterp": ["AR"],
        "InterpText": ["Laboratory action required for oil sample"],
        "DateSampled": ["2026-01-15"],
        "Fe": [15.0],
        "Cu": [5.0],
    })
    result = analyze_frames(ar_sos, telemetry_raw=None, work_orders_raw=None)
    alerts = result["alerts"]
    assert len(alerts) == 1
    assert alerts.iloc[0]["lab_status"] == "Laboratory Action Required"
    assert alerts.iloc[0]["rule_evidence_strength"] <= 0.40
    assert alerts.iloc[0]["evidence_level"] != "Strong rule match"

    cards = result["maintenance_action_cards"]
    assert (cards["failure_probability_pct"].isna()).all(), "AR alone must never produce a failure probability"
    assert "Not Available" in cards.iloc[0]["predicted_risk_status"]
    assert result["trained_model"] is None, "ML model must not run when operating mode is Alert Management"


def test_6_tier_priority_classification():
    mock_sos = pd.DataFrame({
        "SampleNum": ["S01", "S02", "S03", "S04", "S05", "S06", "S07"],
        "EquipNum":  ["EQ-1", "EQ-2", "EQ-3", "EQ-3", "EQ-4", "EQ-5", "EQ-6"],
        "Compartment": ["ENG", "HYD", "TRANS", "TRANS", "DIFF", "ENG", "HYD"],
        "OverallInterp": ["AR", "AR", "AR", "AR", "AR", "A", "AR"],
        "HighPriority":  ["T",   "T",    "F",    "F",    "F",   "F",   "F"],
        "Status":        ["New", "New",  "New",  "New",  "New", "Closed", "New"],
        "WorkOrderId":   [None,  "WO-10", None,  None,   "WO-20", None,  None],
        "DateSampled":   ["2026-01-10", "2026-01-11", "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "invalid-date"],
    })
    result = analyze_frames(mock_sos)
    alerts = result["alerts"].set_index("sample_number")

    # S01: HighPriority=T, Status=New, no WO -> P1 – Immediate Review
    assert alerts.loc["S01", "priority_tier"] == P1_IMMEDIATE
    assert "HighPriority flag is set" in alerts.loc["S01", "priority_reason"]

    # S02: HighPriority=T, Status=New, has WO -> P1 – WO Tracking
    assert alerts.loc["S02", "priority_tier"] == P1_WO_TRACKING
    assert "Verify its status" in alerts.loc["S02", "priority_reason"]

    # S03 & S04: multiple New AR records for (EQ-3, TRANS) without WO -> P2
    assert alerts.loc["S04", "priority_tier"] == P2_REPEATED
    assert "2 New AR records" in alerts.loc["S04", "priority_reason"]

    # S05: New AR with WorkOrderId -> In Progress
    assert alerts.loc["S05", "priority_tier"] == TIER_IN_PROGRESS

    # S06: Status=Closed -> Closed
    assert alerts.loc["S06", "priority_tier"] == TIER_CLOSED

    # S07: Date quality is separate and does not erase the action priority
    assert alerts.loc["S07", "priority_tier"] == P3_ACTION
    assert alerts.loc["S07", "data_quality_status"] == "Review required"
    assert "Unparseable" in alerts.loc["S07", "data_quality_issue"]


def test_dataset_metrics_exact_counts():
    sos = load_table(ROOT / "data/current/SosFluidSample.xlsx")
    result = analyze_frames(sos, telemetry_raw=None, work_orders_raw=None)
    m = result["dataset_metrics"]

    assert m["unique_assets"] == 520
    assert m["lab_ar_samples"] == 1051
    assert m["high_priority_samples"] == 10
    assert m["new_alerts"] == 929
    assert m["closed_alerts"] == 122
    assert m["wo_linked_samples"] == 648
    assert m["samples_without_wo"] == 403
    assert m["new_alerts_without_wo"] == 364
    assert m["invalid_date_samples"] == 552
    assert m["valid_date_samples"] == 499
    assert m["high_priority_new_without_wo"] == 3
    assert m["high_priority_new_with_wo"] == 6
    assert m["high_priority_closed"] == 1
    assert m["repeated_new_unlinked_rows"] == 94
    assert m["p1_immediate_records"] == 3
    assert m["p1_wo_tracking_records"] == 6
    assert m["p2_multiple_unlinked_records"] == 93
    assert m["p3_action_records"] == 268

    priorities = result["alerts"]["priority_tier"].value_counts()
    assert priorities[P1_IMMEDIATE] == 3
    assert priorities[P1_WO_TRACKING] == 6
    assert priorities[TIER_CLOSED] == 122


def test_ml_not_run_in_alert_management_mode():
    sos = load_table(ROOT / "data/current/SosFluidSample.xlsx")
    result = analyze_frames(sos, telemetry_raw=None, work_orders_raw=None)
    assert result["operating_mode"] == MODE_ALERT
    assert result["trained_model"] is None
    assert (result["maintenance_action_cards"]["failure_probability_pct"].isna()).all()


def test_interp_categories_retain_exact_evidence_sentence():
    mock_sos = pd.DataFrame({
        "SampleNum": ["S01"],
        "EquipNum": ["EQ-1"],
        "Compartment": ["ENGINE"],
        "OverallInterp": ["AR"],
        "InterpText": ["High iron wear detected in sample. Inspect bearing and resample within 10 days."],
        "DateSampled": ["2026-01-10"],
    })
    result = analyze_frames(mock_sos)
    cats = result["interp_categories"]
    assert not cats.empty
    assert cats.iloc[0]["problem_category"] == "Wear Metal – Iron"
    assert cats.iloc[0]["category_evidence"] == "High iron wear detected in sample."
    resample = cats[cats["problem_category"] == "Resample / History"]
    assert len(resample) == 1
    assert resample.iloc[0]["category_evidence"] == "Inspect bearing and resample within 10 days."


def test_llm_insights_contain_mode_header(monkeypatch):
    sos = load_table(ROOT / "data/current/SosFluidSample.xlsx")
    result = analyze_frames(sos)
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-used-without-opt-in")
    insights = generate_llm_insights(result)
    assert "Maintenance Triage Summary" in insights
    assert "Alert Management" in insights
    assert "OverallInterp=AR" in insights


def test_time_only_date_is_preserved_and_flagged():
    prepared = prepare_sos(pd.DataFrame({
        "SampleNum": ["S01"],
        "EquipNum": ["EQ-1"],
        "Compartment": ["ENG"],
        "OverallInterp": ["AR"],
        "DateSampled": ["00:00:00"],
    }))
    assert pd.isna(prepared.iloc[0]["sample_date"])
    assert prepared.iloc[0]["sample_date_raw"] == "00:00:00"
    assert prepared.iloc[0]["date_quality_issue"] == "Time-only value; calendar date is missing"


def test_invalid_date_never_suppresses_high_priority_action():
    result = analyze_frames(pd.DataFrame({
        "SampleNum": ["S01"],
        "EquipNum": ["EQ-1"],
        "Compartment": ["ENG"],
        "OverallInterp": ["AR"],
        "HighPriority": ["T"],
        "SampleStatusNew": ["New"],
        "WorkOrderId": [None],
        "DateSampled": ["00:00:00"],
    }))
    alert = result["alerts"].iloc[0]
    assert alert["priority_tier"] == P1_IMMEDIATE
    assert alert["data_quality_status"] == "Review required"


def test_training_labels_are_never_manufactured_from_sos_severity():
    sos = prepare_sos(pd.DataFrame({
        "SampleNum": ["S01", "S02", "S03"],
        "EquipNum": ["EQ-1", "EQ-1", "EQ-1"],
        "Compartment": ["ENG", "ENG", "ENG"],
        "OverallInterp": ["AR", "AR", "AR"],
        "DateSampled": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "Fe": [10, 20, 30],
    }))
    telemetry, _ = prepare_telemetry(None)
    assert build_training_table(sos, telemetry, None).empty

    work_orders = prepare_work_orders(pd.DataFrame({
        "WorkOrderId": ["WO-1", "WO-2"],
        "EquipmentId": ["EQ-1", "EQ-1"],
        "Component": ["ENG", "ENG"],
        "OpenedDate": ["2026-01-10", "2026-02-15"],
        "WorkOrderType": ["CORRECTIVE", "PREVENTIVE"],
        "FailureConfirmed": [1, 0],
    }))
    labelled = build_training_table(sos, telemetry, work_orders, horizon_days=30)
    assert len(labelled) == 3
    assert labelled["corrective_wo_within_horizon"].eq(1).all()
    assert labelled["corrective_wo_within_horizon"].nunique() == 1


def test_small_example_wo_file_does_not_unlock_prediction():
    result = analyze_frames(
        load_table(ROOT / "data/current/SosFluidSample.xlsx"),
        load_table(ROOT / "data/current/TelematicDataSample.xlsx"),
        load_table(ROOT / "data/current/WorkOrderSample.csv"),
    )
    gate = result["readiness"].set_index("gate")
    assert gate.loc["explicit_wo_outcomes", "status"] == "BLOCKED"
    assert result["operating_mode"] == MODE_ALERT
    assert not result["predictive_risk_enabled"]


def test_right_censored_samples_are_not_labelled_negative():
    sos = prepare_sos(pd.DataFrame({
        "SampleNum": ["S01"],
        "EquipNum": ["EQ-1"],
        "Compartment": ["ENG"],
        "OverallInterp": ["A"],
        "DateSampled": ["2026-01-20"],
        "Fe": [10],
    }))
    telemetry, _ = prepare_telemetry(None)
    work_orders = prepare_work_orders(pd.DataFrame({
        "WorkOrderId": ["WO-1"],
        "EquipmentId": ["EQ-1"],
        "Component": ["ENG"],
        "OpenedDate": ["2026-01-25"],
        "WorkOrderType": ["PREVENTIVE"],
        "FailureConfirmed": [0],
    }))
    assert build_training_table(sos, telemetry, work_orders, horizon_days=30).empty


def test_model_training_uses_separate_chronological_calibration_block():
    rows = 90
    training = pd.DataFrame({
        "sample_number": [f"S{i:03d}" for i in range(rows)],
        "asset_id": [f"EQ-{i % 10}" for i in range(rows)],
        "sample_date": pd.date_range("2025-01-01", periods=rows, freq="D"),
        "machine_model": ["MODEL-A" if i % 2 else "MODEL-B" for i in range(rows)],
        "component": ["ENG" if i % 3 else "HYD" for i in range(rows)],
        "equipment_hours": np.arange(rows) * 10.0,
        "fluid_hours": np.arange(rows) % 20,
        "corrective_wo_within_horizon": [i % 2 for i in range(rows)],
        "horizon_days": [30] * rows,
    })
    model, leaderboard = train_failure_models(training)
    assert model.metrics["calibration_rows"] >= 8
    assert model.metrics["test_rows"] >= 8
    assert set(leaderboard["model"]) == {"logistic_regression", "random_forest"}
    probabilities = model.predict_proba(training.head(5))
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_cli_outputs_are_utf8_serializable(tmp_path):
    result = run_analysis(
        ROOT / "data/current/SosFluidSample.xlsx",
        ROOT / "data/current/TelematicDataSample.xlsx",
        output_dir=tmp_path,
    )
    assert result["operating_mode"] == MODE_ALERT
    report = (tmp_path / "ai_insights.md").read_text(encoding="utf-8")
    assert "Maintenance Triage Summary" in report
    assert (tmp_path / "analysis_summary.json").exists()
    assert (tmp_path / "model_readiness.csv").exists()


def test_presentation_uses_current_mode_and_unique_cases():
    result = analyze_frames(
        load_table(ROOT / "data/current/SosFluidSample.xlsx"),
        load_table(ROOT / "data/current/TelematicDataSample.xlsx"),
    )
    mode = build_mode_summary(result, 90)
    cases = build_case_table(result, 90)
    fleet = build_fleet_summary(result, cases, 90)
    assert mode["title"] == "Alert Management active"
    assert not mode["prediction_enabled"]
    assert cases["case_id"].is_unique
    assert len(cases) < len(result["sos"])
    assert fleet["immediate_action"] == 9
    assert fleet["samples"] == 1051


def test_streamlit_dashboard_smoke_and_current_mode():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["P1 cases today"] == "9"
    assert metrics["Evidence confidence"] == "Low"
    view = next(item for item in app.radio if item.label == "View")
    assert view.options == ["Fleet Overview", "Action Queue", "Asset Analysis", "Model Validation", "Data Quality"]
    assert next(item for item in app.select_slider if item.label == "Prediction horizon (days)").value == 30
    assert any("Alert Management active" in item.value for item in app.markdown)


def test_streamlit_matched_demo_runs_prediction_and_validation_page():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    next(item for item in app.radio if item.label == "Data source").set_value("Matched prediction demo")
    app.run(timeout=180)
    assert not app.exception
    assert any("Failure Prediction active" in item.value for item in app.markdown)
    assert any("synthetic" in item.value.lower() for item in app.warning)
    next(item for item in app.radio if item.label == "View").set_value("Model Validation")
    app.run(timeout=180)
    assert not app.exception
    assert any(header.value == "Model Validation" for header in app.header)
    assert build_validation_summary(app.session_state["analysis"]) is not None
