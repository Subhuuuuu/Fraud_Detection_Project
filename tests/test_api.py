
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

VALID_TRANSACTION = {
    "step": 10,
    "type": "TRANSFER",
    "amount": 5000,
    "oldbalanceOrg": 10000,
    "newbalanceOrig": 5000,
    "oldbalanceDest": 2000,
    "newbalanceDest": 7000,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_valid_prediction():
    response = client.post("/predict", json=VALID_TRANSACTION)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["result"] in ["Fraud", "Legitimate"]
    assert 0 <= data["fraud_probability"] <= 1


def test_missing_field():
    data = VALID_TRANSACTION.copy()
    del data["amount"]

    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_wrong_data_type():
    data = VALID_TRANSACTION.copy()
    data["amount"] = "not-a-number"

    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_negative_amount():
    data = VALID_TRANSACTION.copy()
    data["amount"] = -100

    response = client.post("/predict", json=data)
    assert response.status_code == 422


def test_unseen_transaction_type():
    data = VALID_TRANSACTION.copy()
    data["type"] = "CRYPTO"

    response = client.post("/predict", json=data)
    assert response.status_code == 422
