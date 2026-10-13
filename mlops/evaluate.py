
import mlflow
import pandas as pd

TRACKING_URI = (
    "sqlite:////home/azureuser/Fraud_Detection_Project/mlflow.db"
)
EXPERIMENT_NAME = "Fraud_Detection_Experiments"

mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.MlflowClient()

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError(f"Experiment not found: {EXPERIMENT_NAME}")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
)

required_metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
]

records = []

for run in runs:
    metrics = run.data.metrics

    # Exclude dummy or incomplete runs.
    if not all(key in metrics for key in required_metrics):
        continue

    records.append({
        "run_name": run.data.tags.get("mlflow.runName", ""),
        "run_id": run.info.run_id,
        "model_type": run.data.params.get("model_type", "Unknown"),
        **{key: metrics[key] for key in required_metrics},
    })

if not records:
    print("No completed runs contain all required metrics.")
else:
    results = pd.DataFrame(records)
    results = results.sort_values(
        by=["recall", "pr_auc", "f1_score"],
        ascending=False,
    )

    print("\nCompleted runs with all required metrics:")
    print(results.to_string(index=False))
