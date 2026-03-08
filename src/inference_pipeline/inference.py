"""
Inference pipeline: raw input → preprocess → feature engineering → align schema → predict.

Takes raw data (same schema as holdout.csv), applies saved encoders, and returns predictions.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

from src.feature_pipeline.preprocess import clean_and_merge, drop_duplicates, remove_outliers
from src.feature_pipeline.feature_engineering import add_date_features, drop_unused_columns

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
DEFAULT_FREQ_ENCODER = PROJECT_ROOT / "models" / "freq_encoder.pkl"
DEFAULT_TARGET_ENCODER = PROJECT_ROOT / "models" / "target_encoder.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

print("📂 Inference using project root:", PROJECT_ROOT)

# Load expected feature columns from training data (used for schema alignment)
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "price"]
else:
    TRAIN_FEATURE_COLUMNS = None


def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    freq_encoder_path: Path | str = DEFAULT_FREQ_ENCODER,
    target_encoder_path: Path | str = DEFAULT_TARGET_ENCODER,
) -> pd.DataFrame:
    df = clean_and_merge(input_df)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    if "date" in df.columns:
        df = add_date_features(df)

    if Path(freq_encoder_path).exists() and "zipcode" in df.columns:
        freq_map = load(freq_encoder_path)
        df["zipcode_freq"] = df["zipcode"].map(freq_map).fillna(0)
        df = df.drop(columns=["zipcode"], errors="ignore")

    if Path(target_encoder_path).exists() and "city_full" in df.columns:
        target_encoder = load(target_encoder_path)
        df["city_full_encoded"] = target_encoder.transform(df["city_full"])
        df = df.drop(columns=["city_full"], errors="ignore")

    df, _ = drop_unused_columns(df.copy(), df.copy())

    y_true = None
    if "price" in df.columns:
        y_true = df["price"].tolist()
        df = df.drop(columns=["price"])

    # Align to training schema
    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    model = load(model_path)
    preds = model.predict(df)

    out = df.copy()
    out["predicted_price"] = preds
    if y_true is not None:
        out["actual_price"] = y_true

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on new housing data (raw).")
    parser.add_argument("--input", type=str, required=True, help="Path to input RAW CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save predictions CSV")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained model file")
    parser.add_argument("--freq_encoder", type=str, default=str(DEFAULT_FREQ_ENCODER), help="Path to frequency encoder pickle")
    parser.add_argument("--target_encoder", type=str, default=str(DEFAULT_TARGET_ENCODER), help="Path to target encoder pickle")

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model,
        freq_encoder_path=args.freq_encoder,
        target_encoder_path=args.target_encoder,
    )

    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")
