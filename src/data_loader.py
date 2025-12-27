import pandas as pd
from pathlib import Path


def load_crypto_data(csv_path: str) -> pd.DataFrame:
    """
    Load cryptocurrency OHLCV + market_cap data.

    Expected columns (can be adapted):
    - date, symbol, open, high, low, close, volume, market_cap
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    # basic cleaning
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df
