
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "Fraud_Detection_Experiments"
MODEL_NAME = "FraudDetectionRandomForest"
MODEL_VERSION = "1"

FRESH_RUN_ID = "f802ac97356b45f48124f3b6295c7d48"

REQUIRED_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: {EXPERIMENT_NAME}"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )

    rows = []

    for run in runs:
        metrics = run.data.metrics

        if not all(name in metrics for name in REQUIRED_METRICS):
            continue

        rows.append({
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "run_id": run.info.run_id,
            "model_type": run.data.params.get("model_type", ""),
            **{
                name: metrics[name]
                for name in REQUIRED_METRICS
            },
        })

    if not rows:
        raise RuntimeError(
            "No completed runs contain all required metrics."
        )

    results = pd.DataFrame(rows)

    # Sort newest runs first, then remove duplicate metric records.
    results = results.sort_values(
        "run_id",
        ascending=False,
    )

    results = results.drop_duplicates(
        subset=["run_name", *REQUIRED_METRICS],
        keep="first",
    )

    results = results.sort_values(
        ["pr_auc", "recall", "f1_score"],
        ascending=False,
    ).reset_index(drop=True)

    output_path = ROOT / "results" / "mlflow_run_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("\nMLflow experiment comparison:")
    print(results.to_string(index=False))
    print(f"\nComparison saved to: {output_path}")

    # Select the specific fresh evaluation run.
    selected_rows = results[
        results["run_id"] == FRESH_RUN_ID
    ]

    if selected_rows.empty:
        raise RuntimeError(
            "The fresh evaluation run was not found in the "
            "comparison report. Check FRESH_RUN_ID."
        )

    selected = selected_rows.iloc[0]

    client.set_model_version_tag(
        MODEL_NAME,
        MODEL_VERSION,
        "selected_run_id",
        selected["run_id"],
    )
    client.set_model_version_tag(
        MODEL_NAME,
        MODEL_VERSION,
        "selection_metric",
        "pr_auc",
    )
    client.set_model_version_tag(
        MODEL_NAME,
        MODEL_VERSION,
        "selection_status",
        "selected_pending_independent_validation",
    )

    print("\nSelected model:")
    print(f"Model: {MODEL_NAME}")
    print(f"Version: {MODEL_VERSION}")
    print(f"Evaluation run: {selected['run_id']}")
    print(f"PR-AUC: {selected['pr_auc']:.6f}")
    print("Validation: pending independent validation")
    print("Production promotion: not performed")


if __name__ == "__main__":
    main()
