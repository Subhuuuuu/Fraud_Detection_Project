
from pathlib import Path
import warnings

import mlflow
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MODEL_URI = "models:/FraudDetectionRandomForest/1"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    print("Loading registered model...")
    model = mlflow.pyfunc.load_model(MODEL_URI)

    expected = list(model.metadata.signature.inputs.input_names())
    print(f"Expected feature count: {len(expected)}")
    print("Expected features:", expected)

    # Synthetic input for a smoke test, not a real transaction.
    sample = pd.DataFrame([[0.0] * len(expected)], columns=expected)

    print("\nRunning a synthetic prediction...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prediction = model.predict(sample)

    print("Prediction output:", prediction)

    version_warnings = [
        warning for warning in caught
        if issubclass(warning.category, InconsistentVersionWarning)
    ]

    if version_warnings:
        print("\nCOMPATIBILITY WARNING:")
        for warning in version_warnings:
            print(warning.message)
        print("Do not treat this model as production-ready until versions are aligned.")
    else:
        print("\nNo scikit-learn version warning was raised during this prediction.")

    print("\nSmoke test completed.")


if __name__ == "__main__":
    main()
