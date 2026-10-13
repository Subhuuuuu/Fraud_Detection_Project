
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
REPORT_PATH = ROOT / "results" / "mlflow_run_comparison.csv"

MODEL_NAME = "FraudDetectionRandomForest"
MODEL_VERSION = "1"
SELECTED_RUN_ID = "f802ac97356b45f48124f3b6295c7d48"

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Comparison report not found: {REPORT_PATH}"
        )

    results = pd.read_csv(REPORT_PATH)

    required_columns = {"run_id", "run_name", *METRICS}
    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(
            f"Missing report columns: {sorted(missing)}"
        )

    selected = results[results["run_id"] == SELECTED_RUN_ID]

    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one selected evaluation run in the report."
        )

    values = selected.iloc[0][METRICS].astype(float)

    if values.isna().any():
        raise ValueError("Selected run contains missing metrics.")

    if not values.between(0, 1).all():
        raise ValueError("Metric values must be between 0 and 1.")

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    version = client.get_model_version(
        MODEL_NAME, MODEL_VERSION
    )

    if version.tags.get("selected_run_id") != SELECTED_RUN_ID:
        raise ValueError(
            "Registry selection metadata does not match the report."
        )

    print("Automated evaluation checks passed.")
    print(f"Model: {MODEL_NAME} v{MODEL_VERSION}")
    print(f"Selected run: {SELECTED_RUN_ID}")
    print(f"PR-AUC: {values['pr_auc']:.6f}")
    print("Independent validation: still pending")


if __name__ == "__main__":
    main()
