"""
End-to-end training script.

Usage (from repo root):
    python -m src.train
"""

from pathlib import Path
import pandas as pd

from data_loader import load_crypto_data
from preprocess import clean_data, time_based_split, scale_features
from features import build_feature_set
from model import train_volatility_model, evaluate_model


DATA_PATH = Path("data/crypto_prices.csv")
TARGET_COL = "target_volatility"


def main():
    # 1. Load
    df = load_crypto_data(str(DATA_PATH))

    # 2. Clean
    df = clean_data(df)

    # 3. Feature engineering + target
    df_feat = build_feature_set(df, target_col=TARGET_COL)

    # 4. Select features
    feature_cols = [
        col for col in df_feat.columns
        if col not in ["date", "symbol", TARGET_COL]
    ]
    X = df_feat[feature_cols]
    y = df_feat[TARGET_COL]

    # 5. Time-based split
    df_for_split = df_feat[["date", *feature_cols, TARGET_COL]]
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(
        df_for_split, target_col=TARGET_COL, test_size=0.2, val_size=0.1
    )

    # drop date column from features
    X_train = X_train.drop(columns=["date"])
    X_val = X_val.drop(columns=["date"])
    X_test = X_test.drop(columns=["date"])

    # 6. Scale
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # 7. Train model
    model = train_volatility_model(X_train_s, y_train)

    # 8. Evaluate
    val_res = evaluate_model(model, X_val_s, y_val)
    test_res = evaluate_model(model, X_test_s, y_test)

    print("Validation metrics:")
    print(f"  RMSE: {val_res.rmse:.6f}")
    print(f"  MAE : {val_res.mae:.6f}")
    print(f"  R^2 : {val_res.r2:.4f}")

    print("\nTest metrics:")
    print(f"  RMSE: {test_res.rmse:.6f}")
    print(f"  MAE : {test_res.mae:.6f}")
    print(f"  R^2 : {test_res.r2:.4f}")

    # 9. Save artifacts
    import joblib
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    joblib.dump(model, artifacts_dir / "volatility_model.pkl")
    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    joblib.dump(feature_cols, artifacts_dir / "feature_cols.pkl")
    print(f"\nArtifacts saved in: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
