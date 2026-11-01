FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY diabetes_api.py .

EXPOSE 8000

CMD ["uvicorn", "diabetes_api:app", "--host", "0.0.0.0", "--port", "8000"]
