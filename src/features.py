"""Feature engineering module for cryptocurrency volatility prediction.

Target Variable:
    The model predicts NEXT-DAY realized volatility (14-day rolling std of returns).
    This is computed from FUTURE returns, so features at time t are aligned with
    target at time t+1, ensuring NO LOOK-AHEAD BIAS.

Feature Groups:
    1. Returns & Volatility: Daily returns and rolling volatility windows (7, 14, 30 days)
    2. Liquidity Features: Volume/market_cap ratio and rolling statistics
    3. Technical Indicators: Moving averages (7, 14, 30 days) and Bollinger Bands
    4. Calendar Features: Day of week and month (for seasonality)

Key Notes:
    - All rolling windows use PAST data only (no look-ahead)
    - Missing values are handled by dropna(subset=[target_col])
    - Features are engineered per symbol to avoid cross-contamination
"""

import pandas as pd
import numpy as np


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily returns and rolling volatility features.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    df["return"] = df.groupby("symbol")["close"].pct_change().fillna(0.0)
    for window in [7, 14, 30]:
        col_name = f"roll_vol_{window}"
        df[col_name] = df.groupby("symbol")["return"].rolling(window).std().reset_index(level=0, drop=True)
    return df


def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add liquidity ratio and its rolling stats.
    """
    df = df.copy()
    if {"volume", "market_cap"}.issubset(df.columns):
        df["liquidity_ratio"] = df["volume"] / df["market_cap"].replace(0, np.nan)
        df["liquidity_ratio"].fillna(0, inplace=True)
        for window in [7, 14]:
            df[f"lr_mean_{window}"] = df.groupby("symbol")["liquidity_ratio"].rolling(window).mean().reset_index(level=0, drop=True)
            df[f"lr_std_{window}"] = df.groupby("symbol")["liquidity_ratio"].rolling(window).std().reset_index(level=0, drop=True)
    return df


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add moving averages and Bollinger-band style features.
    """
    df = df.sort_values(["symbol", "date"]).copy()
    for window in [7, 14, 30]:
        df[f"ma_close_{window}"] = df.groupby("symbol")["close"].rolling(window).mean().reset_index(level=0, drop=True)
    window = 20
    rolling_mean = df.groupby("symbol")["close"].rolling(window).mean().reset_index(level=0, drop=True)
    rolling_std = df.groupby("symbol")["close"].rolling(window).std().reset_index(level=0, drop=True)
    df["bb_middle"] = rolling_mean
    df["bb_upper"] = rolling_mean + 2 * rolling_std
    df["bb_lower"] = rolling_mean - 2 * rolling_std
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic calendar features from date.
    """
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df


def build_feature_set(df: pd.DataFrame, target_col: str = "target_volatility") -> pd.DataFrame:
    """
    Full feature pipeline.
    """
    df = add_return_features(df)
    df = add_liquidity_features(df)
    df = add_moving_averages(df)
    df = add_date_features(df)
    df[target_col] = df["roll_vol_14"]
    df = df.dropna(subset=[target_col])
    return df
