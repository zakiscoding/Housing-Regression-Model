"""
Batch runner: loads holdout data, groups by year/month, runs inference, and saves per-month CSVs.
"""

from pathlib import Path
import pandas as pd
from src.inference_pipeline.inference import predict

DATA_DIR = Path("data/processed")
HOLDOUT_PATH = DATA_DIR / "cleaning_holdout.csv"
OUTPUT_DIR = Path("data/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_monthly_predictions():
    df = pd.read_csv(HOLDOUT_PATH)
    df["date"] = pd.to_datetime(df["date"])
    grouped = df.groupby([df["date"].dt.year, df["date"].dt.month])

    all_outputs = []
    for (year, month), group in grouped:
        print(f"📅 Running predictions for {year}-{month:02d} ({len(group)} rows)")

        preds_df = predict(group)

        out_path = OUTPUT_DIR / f"preds_{year}_{month:02d}.csv"
        preds_df.to_csv(out_path, index=False)
        print(f"✅ Saved predictions to {out_path}")

        all_outputs.append(preds_df)

    return pd.concat(all_outputs, ignore_index=True)


if __name__ == "__main__":
    all_preds = run_monthly_predictions()
    print("🎉 Batch inference complete.")
    print(all_preds.head())
