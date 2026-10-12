
from pathlib import Path
import pickle

import pandas as pd
from fastapi import FastAPI

from api.schemas import TransactionRequest, PredictionResponse


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_fraud_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

app = FastAPI(title="Fraud Detection API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    data = transaction.model_dump()

    row = {
        "step": data["step"],
        "amount": data["amount"],
        "oldbalanceOrg": data["oldbalanceOrg"],
        "newbalanceOrig": data["newbalanceOrig"],
        "oldbalanceDest": data["oldbalanceDest"],
        "newbalanceDest": data["newbalanceDest"],
        "balance_change_orig": (
            data["oldbalanceOrg"]
            - data["amount"]
            - data["newbalanceOrig"]
        ),
        "balance_change_dest": (
            data["oldbalanceDest"]
            + data["amount"]
            - data["newbalanceDest"]
        ),
        "orig_balance_zero_after": int(
            data["newbalanceOrig"] == 0
        ),
        "dest_balance_zero_before": int(
            data["oldbalanceDest"] == 0
        ),
    }

    for transaction_type in [
        "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"
    ]:
        row[f"type_{transaction_type}"] = int(
            data["type"] == transaction_type
        )

    features = pd.DataFrame([row])

    # Match the exact feature order used during model training.
    expected_columns = list(model.feature_names_in_)
    features = features.reindex(
        columns=expected_columns, fill_value=0
    )

    probability = float(model.predict_proba(features)[0, 1])
    prediction = int(probability >= 0.5)

    return PredictionResponse(
        prediction=prediction,
        result="Fraud" if prediction else "Legitimate",
        fraud_probability=probability,
    )
