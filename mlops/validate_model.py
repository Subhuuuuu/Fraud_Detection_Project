
from pathlib import Path
import warnings

import mlflow
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MODEL_URI = "models:/FraudDetectionRandomForest/1"
DATA_PATH = ROOT / "data" / "transactions.csv"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    print("Loading registered model...")
    model = mlflow.pyfunc.load_model(MODEL_URI)

    expected = list(model.metadata.signature.inputs.input_names())
    print("Expected feature count:", len(expected))
    print("Expected features:", expected)

    # First perform a synthetic-input smoke test.
    sample = pd.DataFrame(
        [[0.0] * len(expected)],
        columns=expected,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prediction = model.predict(sample)

    print("Synthetic prediction:", prediction)

    version_warnings = [
        w for w in caught
        if issubclass(w.category, InconsistentVersionWarning)
    ]

    if version_warnings:
        print("WARNING: Model compatibility warning detected.")
        for warning in version_warnings:
            print(warning.message)
    else:
        print("No scikit-learn compatibility warning during prediction.")

    # Validate data availability and feature compatibility.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("\nChecking dataset columns...")
    data_columns = set(pd.read_csv(DATA_PATH, nrows=0).columns)

    required = {
        "step", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "isFraud",
    }

    missing = required - data_columns
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    print("Required source columns are present.")
    print(
        "\nNOTE: This is a smoke test and data-schema check. "
        "It does not establish independent model validation."
    )


if __name__ == "__main__":
    main()
