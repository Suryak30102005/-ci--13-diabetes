
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from prefect import flow, task

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "diabetes_progression.csv"

TARGET = "DiseaseProgression"

mlflow.set_tracking_uri("http://127.0.0.1:5000")


@task
def load_data():
    df = pd.read_csv(DATA_FILE)

    print("Dataset loaded:", df.shape)
    return df


@task
def validate_data(df):
    if df.empty:
        raise ValueError("Dataset is empty")

    if TARGET not in df.columns:
        raise ValueError("Target column missing")

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    print("Data quality validation passed")

    return df


@task
def train_evaluate(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

    mlflow.set_experiment("EXP13_Diabetes_Progression")

    with mlflow.start_run(run_name="prefect-ridge") as run:

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("r2", float(r2))

        mlflow.log_param("algorithm", "Ridge")
        mlflow.log_param("alpha", 1.0)
        mlflow.log_param("random_state", 42)

        if not np.isfinite(rmse) or rmse > 70:
            raise ValueError(
                f"Quality gate failed: RMSE = {rmse:.4f}"
            )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=X_train.head(3),
            serialization_format="cloudpickle"
        )

    print("Model evaluation completed")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "run_id": run.info.run_id
    }


@task
def quality_gate(metrics):
    print("Quality gate passed")
    print("Approved run:", metrics["run_id"])

    return metrics


@flow(name="diabetes-training-pipeline")
def diabetes_training_pipeline():
    df = load_data()
    clean_df = validate_data(df)
    metrics = train_evaluate(clean_df)
    return quality_gate(metrics)


if __name__ == "__main__":
    diabetes_training_pipeline()
