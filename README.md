# EXP-13: Diabetes Disease Progression MLOps

**Name:** Gopi Krishna  
**Registration Number:** 231FA04310

## Aim
Implement an end-to-end MLOps pipeline using Scikit-learn, MLflow, Prefect, FastAPI, Docker, Prometheus, and Evidently.

## Dataset
- Scikit-learn Diabetes Dataset
- 442 records
- 10 input features
- 353 training and 89 testing records

## Model Results

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Ridge | 42.8120 | 53.7775 | 0.4541 |
| Linear Regression | 42.7941 | 53.8534 | 0.4526 |
| Random Forest | 44.1712 | 54.3984 | 0.4415 |
| HistGradientBoosting | 45.0227 | 57.0080 | 0.3866 |

## Champion Model
- Registered model: DiabetesProgressionModel
- Algorithm: Ridge Regression
- Version: 1
- Alias: champion

## Implementation
1. Dataset preprocessing
2. Model training and comparison
3. MLflow experiment tracking
4. Prefect training orchestration
5. FastAPI model serving
6. Docker containerization
7. Prometheus monitoring
8. Evidently simulated data drift
9. Prediction error analysis

## API
- GET /health
- POST /predict
- GET /metrics

## Docker
Build: docker build -t diabetes-prediction-api:1.0 .

Swagger URL: http://127.0.0.1:8003/docs

## Notes
Model artifacts and the MLflow registry must be available separately.
This dataset is for regression benchmarking, not clinical diagnosis.
