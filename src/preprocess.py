import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and ensure basic consistency.
    """
    # drop rows with missing critical fields
    df = df.dropna(subset=["date", "symbol", "close"])

    # forward/backward fill numeric columns per symbol
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = (
        df.groupby("symbol")[numeric_cols]
        .apply(lambda g: g.ffill().bfill())
        .reset_index(level=0, drop=True)
    )

    # basic sanity check
    if {"high", "low", "open", "close"}.issubset(df.columns):
        mask = (df["low"] <= df["open"]) & (df["low"] <= df["close"]) & \
               (df["high"] >= df["open"]) & (df["high"] >= df["close"])
        df = df[mask]

    df = df.drop_duplicates(subset=["symbol", "date"])
    return df


def scale_features(X_train, X_val, X_test):
    """
    Scale numerical features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def time_based_split(df: pd.DataFrame, target_col: str, test_size: float = 0.2, val_size: float = 0.1):
    """
    Simple time-based split: sort by date and split into train/val/test.
    """
    df = df.sort_values("date")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    n = len(df)
    test_n = int(n * test_size)
    train_temp_n = n - test_n

    X_train_temp, X_test = X.iloc[:train_temp_n], X.iloc[train_temp_n:]
    y_train_temp, y_test = y.iloc[:train_temp_n], y.iloc[train_temp_n:]

    val_n = int(len(X_train_temp) * val_size)
    X_train = X_train_temp.iloc[:-val_n]
    X_val = X_train_temp.iloc[-val_n:]
    y_train = y_train_temp.iloc[:-val_n]
    y_val = y_train_temp.iloc[-val_n:]

    return X_train, X_val, X_test, y_train, y_val, y_test
