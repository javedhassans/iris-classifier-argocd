from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from .model import predict
from .utils import load_model, validate_features


app = FastAPI(title="Iris Classifier", version="0.1.0")

MODEL_PATH = os.getenv("MODEL_PATH", "model/iris_model.pkl")

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run training first.")


class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionResponse(BaseModel):
    class_: str
    class_id: int
    probabilities: dict


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    try:
        features = validate_features([
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ])
        result = predict(model, features)
        return PredictionResponse(
            class_=result["class"],
            class_id=result["class_id"],
            probabilities=result["probabilities"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
