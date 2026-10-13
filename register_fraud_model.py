from pathlib import Path
import pickle
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_fraud_model.pkl"
RESULTS_PATH = ROOT / "results" / "model_comparison.csv"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MODEL_NAME = "FraudDetectionRandomForest"

# Load the existing artifact. Do not retrain or overwrite it.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", InconsistentVersionWarning)
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

results = pd.read_csv(RESULTS_PATH)
match = results[results["Model"] == "Random Forest"]

if match.empty:
    raise ValueError("Random Forest results were not found in model_comparison.csv")

metrics = match.iloc[0]

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Fraud_Detection_Experiments")

# Construct representative input using the model's expected feature names.
columns = list(model.feature_names_in_)
sample = pd.DataFrame([[0.0] * len(columns)], columns=columns)

signature = infer_signature(
    sample,
    model.predict(sample),
)

with mlflow.start_run(run_name="RandomForest_existing_artifact") as run:
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("pipeline_type", "imblearn.pipeline.Pipeline")
    mlflow.log_param("source_artifact", str(MODEL_PATH))
    mlflow.log_param("sklearn_runtime_version", sklearn.__version__)
    mlflow.log_param("feature_count", len(columns))
    mlflow.log_param("metrics_source", "results/model_comparison.csv")
    mlflow.log_param("validation_status", "Existing metrics; version compatibility pending")

    metric_map = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1-score": "f1_score",
        "ROC-AUC": "roc_auc",
        "PR-AUC": "pr_auc",
    }

    for source_name, metric_name in metric_map.items():
        mlflow.log_metric(metric_name, float(metrics[source_name]))

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="fraud_detection_model",
        signature=signature,
        input_example=sample,
        registered_model_name=MODEL_NAME,
        skops_trusted_types=[
        "imblearn.over_sampling._smote.base.SMOTE",
        "imblearn.pipeline.Pipeline",
        "sklearn.metrics._dist_metrics.EuclideanDistance64",
        "sklearn.neighbors._kd_tree.KDTree",
        "sklearn.tree._tree.Tree",
        ],
        pip_requirements=[
            f"mlflow=={mlflow.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"imbalanced-learn=={__import__('imblearn').__version__}",
            f"numpy=={__import__('numpy').__version__}",
            f"pandas=={pd.__version__}",
        ],
    )

    print("Run ID:", run.info.run_id)
    print("Model URI:", model_info.model_uri)
    print("Registered model:", MODEL_NAME)

print("\nRecorded Random Forest metrics:")
for source_name, metric_name in metric_map.items():
    print(f"{source_name}: {float(metrics[source_name]):.6f}")
