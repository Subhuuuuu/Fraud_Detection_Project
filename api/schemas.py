
from typing import Literal
from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    step: int = Field(ge=1)
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(gt=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    result: str
    fraud_probability: float
