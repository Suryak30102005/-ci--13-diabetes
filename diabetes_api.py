
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# MLflow configuration
mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)

# Load registered champion model
model = mlflow.sklearn.load_model(
    "models:/DiabetesProgressionModel@champion"
)

app = FastAPI(
    title="Diabetes Progression Prediction API",
    version="1.0.0",
    description="MLOps Experiment 13"
)

REQUESTS = Counter(
    "diabetes_prediction_requests_total",
    "Total prediction requests"
)

LATENCY = Histogram(
    "diabetes_prediction_latency_seconds",
    "Prediction request processing time"
)


class DiabetesInput(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/")
def home():
    return {
        "message": "Diabetes Progression API",
        "model": "Ridge Regression",
        "experiment": "EXP13"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "DiabetesProgressionModel"
    }


@app.post("/predict")
def predict(data: DiabetesInput):

    REQUESTS.inc()

    row = pd.DataFrame([
        data.model_dump()
    ])

    with LATENCY.time():
        prediction = float(
            model.predict(row)[0]
        )

    return {
        "predicted_disease_progression": prediction
    }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
