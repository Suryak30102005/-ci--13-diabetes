
from pathlib import Path
import json

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "diabetes_progression.csv"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["DiseaseProgression"])
y = df["DiseaseProgression"]

# Use the original test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# Load existing champion model from MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.sklearn.load_model(
    "models:/DiabetesProgressionModel@champion"
)

# Generate predictions
predictions = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Create individual prediction records
results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": predictions
})

results["Error"] = results["Predicted"] - results["Actual"]
results["Absolute_Error"] = results["Error"].abs()

# Save CSV output
results.to_csv(
    REPORT_DIR / "diabetes_prediction_errors.csv",
    index=False
)

summary = {
    "test_records": len(results),
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

with open(
    REPORT_DIR / "diabetes_performance_summary.json",
    "w"
) as file:
    json.dump(summary, file, indent=4)

# Actual vs predicted visualization
plt.figure(figsize=(7, 5))

plt.scatter(
    results["Actual"],
    results["Predicted"],
    alpha=0.7
)

low = min(results["Actual"].min(), results["Predicted"].min())
high = max(results["Actual"].max(), results["Predicted"].max())

plt.plot([low, high], [low, high], "--")

plt.xlabel("Actual Disease Progression")
plt.ylabel("Predicted Disease Progression")
plt.title("Actual vs Predicted — Ridge Regression")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "diabetes_actual_vs_predicted.png",
    dpi=150
)

plt.close()

print("\nDIABETES MODEL PERFORMANCE")
print("--------------------------")
print("Test records:", len(results))
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2:", round(r2, 4))

print("\nReports generated successfully")
print("Location:", REPORT_DIR)
