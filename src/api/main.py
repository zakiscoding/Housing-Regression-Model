"""
FastAPI service for housing price predictions.
Downloads model and training features from S3 on first start, then serves predictions.
"""

from fastapi import FastAPI
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import boto3, os
from joblib import load

S3_BUCKET = os.getenv("S3_BUCKET", "model-regression-data")
REGION = os.getenv("AWS_REGION", "us-east-2")
s3 = boto3.client("s3", region_name=REGION)


def load_from_s3(key, local_path):
    """Download file from S3 if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        print(f"📥 Downloading {key} from S3…")
        s3.download_file(S3_BUCKET, key, str(local_path))
    return str(local_path)


MODEL_PATH = Path(load_from_s3("models/xgb_best_model.pkl", "models/xgb_best_model.pkl"))

# Load model once at startup and derive expected features from booster
_model = load(MODEL_PATH)
FEATURE_NAMES = _model.get_booster().feature_names

app = FastAPI(title="Housing Regression API")


@app.get("/")
def root():
    return {"message": "Housing Regression API is running"}


@app.get("/health")
def health():
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH), "status": "healthy"}
    if not MODEL_PATH.exists():
        status["status"] = "unhealthy"
        status["error"] = "Model not found"
    else:
        status["n_features_expected"] = len(FEATURE_NAMES) if FEATURE_NAMES else 0
    return status


@app.post("/predict")
def predict_batch(data: List[dict]):
    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided"}

    y_true = df.pop("price").tolist() if "price" in df.columns else None

    # Align to exact features the model was trained on
    df = df.reindex(columns=FEATURE_NAMES, fill_value=0)

    preds = _model.predict(df).tolist()
    resp = {"predictions": preds}
    if y_true is not None:
        resp["actuals"] = y_true
    return resp


from src.batch.run_monthly import run_monthly_predictions


@app.post("/run_batch")
def run_batch():
    preds = run_monthly_predictions()
    return {
        "status": "success",
        "rows_predicted": int(len(preds)),
        "output_dir": "data/predictions/"
    }


@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    pred_dir = Path("data/predictions")
    files = sorted(pred_dir.glob("preds_*.csv"))
    if not files:
        return {"error": "No predictions found"}

    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records")
    }
