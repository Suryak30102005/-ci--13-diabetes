
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

# Project paths
ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "diabetes_progression.csv"

REPORT_FILE = ROOT / "reports" / "diabetes_drift_report.html"

# Load dataset
df = pd.read_csv(DATA_FILE)

# Use only model input features
X = df.drop(columns=["DiseaseProgression"])

# Reproduce original train-test split
reference, current = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)

reference = reference.copy()
current = current.copy()

print("Reference records:", len(reference))
print("Current records:", len(current))

# Simulate data drift in incoming records
current["bmi"] = current["bmi"] + 0.15
current["bp"] = current["bp"] + 0.12

# All diabetes features are numerical
schema = DataDefinition(
    numerical_columns=list(X.columns)
)

reference_dataset = Dataset.from_pandas(
    reference,
    data_definition=schema
)

current_dataset = Dataset.from_pandas(
    current,
    data_definition=schema
)

# Generate Evidently report
report = Report([DataDriftPreset()])

snapshot = report.run(
    current_dataset,
    reference_dataset
)

# Save report
REPORT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

snapshot.save_html(str(REPORT_FILE))

print("\nEVIDENTLY DATA DRIFT REPORT")
print("Status: Generated successfully")
print("Simulated drift: bmi, bp")
print("Report:", REPORT_FILE)
