
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "results" / "mlflow_run_comparison.csv"

REQUIRED_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]


def test_comparison_report_exists():
    assert REPORT_PATH.is_file()


def test_comparison_report_has_required_columns():
    results = pd.read_csv(REPORT_PATH)

    required = {"run_id", "run_name", *REQUIRED_METRICS}

    assert required.issubset(results.columns)


def test_selected_evaluation_run_is_present():
    results = pd.read_csv(REPORT_PATH)

    selected_run_id = "f802ac97356b45f48124f3b6295c7d48"

    assert (results["run_id"] == selected_run_id).sum() == 1


def test_selected_metrics_are_valid():
    results = pd.read_csv(REPORT_PATH)

    selected = results.loc[
        results["run_id"] == "f802ac97356b45f48124f3b6295c7d48"
    ]

    assert len(selected) == 1

    values = selected.iloc[0][REQUIRED_METRICS]

    assert values.notna().all()
    assert values.between(0, 1).all()
