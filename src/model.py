from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


@dataclass
class VolatilityModelResult:
    model: RandomForestRegressor
    rmse: float
    mae: float
    r2: float


def train_volatility_model(X_train, y_train, n_estimators: int = 200, max_depth: int = None):
    """
    Train a RandomForest regressor for volatility prediction.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y) -> VolatilityModelResult:
    """
    Evaluate model performance and return metrics.
    """
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return VolatilityModelResult(model=model, rmse=rmse, mae=mae, r2=r2)
