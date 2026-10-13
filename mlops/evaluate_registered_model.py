
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "transactions.csv"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MODEL_NAME = "FraudDetectionRandomForest"
MODEL_VERSION = "1"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    print("Loading transaction data...")
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().copy()

    print("Dataset shape:", df.shape)

    # Reproduce the notebook's feature engineering.
    df["balance_change_orig"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )
    df["balance_change_dest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )
    df["orig_balance_zero_after"] = (
        df["newbalanceOrig"] == 0
    ).astype(int)
    df["dest_balance_zero_before"] = (
        df["oldbalanceDest"] == 0
    ).astype(int)

    y = df["isFraud"].astype(int)
    X = df.drop(
        columns=["isFraud", "isFlaggedFraud", "nameOrig", "nameDest"]
    )
    X = pd.get_dummies(X, columns=["type"], dtype=int)

    print("Loading registered model...")
    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )

    expected_features = list(model.feature_names_in_)
    X = X.reindex(columns=expected_features, fill_value=0)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("Test rows:", len(X_test))
    print("Fraud cases in test set:", int(y_test.sum()))
    print("Generating predictions...")

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
    }

    print("\nEvaluation results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    experiment_name = "Fraud_Detection_Experiments"
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="RandomForest_heldout_evaluation",
    ) as run:
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("model_version", MODEL_VERSION)
        mlflow.log_param("dataset", "data/transactions.csv")
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("threshold", 0.5)
        mlflow.log_metrics(metrics)

        run_id = run.info.run_id

    client.set_model_version_tag(
        MODEL_NAME, MODEL_VERSION, "evaluation_run_id", run_id
    )
    client.set_model_version_tag(
        MODEL_NAME, MODEL_VERSION,
        "validation_status", "metrics_calculated_pending_review"
    )
    client.set_model_version_tag(
        MODEL_NAME, MODEL_VERSION,
        "metrics_source", "fresh_evaluation"
    )

    print("\nEvaluation logged to MLflow.")
    print("Run ID:", run_id)
    print("Model version:", MODEL_VERSION)
    print("Validation remains pending review.")


if __name__ == "__main__":
    main()
