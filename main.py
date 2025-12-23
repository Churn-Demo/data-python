from fastapi import FastAPI, HTTPException, UploadFile, File
from pathlib import Path
from io import StringIO
import pandas as pd

from schemas import PredictByIdRequest
from ml import FEATURE_COLS, load_or_train
from repo import get_customer_row

app = FastAPI(title="Data Service (Demo)")

DATA_PATH = Path("data/churn_demo.csv")
MODEL_PATH = Path("model.joblib")
CUSTOMERS_PATH = Path("data/new_customers.csv")

model = None

@app.on_event("startup")
def startup():
    global model
    model = load_or_train(DATA_PATH, MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "dataset": str(DATA_PATH), "model_loaded": model is not None}

@app.post("/predict_by_id")
def predict_by_id(req: PredictByIdRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    row = get_customer_row(CUSTOMERS_PATH, req.customer_id)
    if row.empty:
        raise HTTPException(status_code=404, detail="Customer not found")

    X = row[FEATURE_COLS]
    proba = float(model.predict_proba(X)[0][1])
    pred = int(proba >= 0.5)

    if proba >= 0.70:
        risk_level = "ALTO"
    elif proba >= 0.40:
        risk_level = "MEDIO"
    else:
        risk_level = "BAJO"

    return {
        "customer_id": req.customer_id,
        "churn_probability": round(proba, 3),
        "prediction": pred,
        "risk_level": risk_level,
        "model_version": "dataset-demo-v1",
        "source": "new_customers"
    }

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    content = (await file.read()).decode("utf-8", errors="ignore")
    df = pd.read_csv(StringIO(content))

    missing = [c for c in (["customer_id"] + FEATURE_COLS) if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    probs = model.predict_proba(df[FEATURE_COLS])[:, 1]
    preds = (probs >= 0.5).astype(int)

    out = df[["customer_id"] + FEATURE_COLS].copy()
    out["churn_probability"] = probs.round(3)
    out["prediction"] = preds
    out["model_version"] = "dataset-demo-v1"
    return out.to_dict(orient="records")
