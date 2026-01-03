# Cryptocurrency Volatility Prediction

End-to-end ML project to predict cryptocurrency volatility using historical OHLCV and market cap data.

## Problem and goals

This project predicts short-term cryptocurrency **volatility** to help traders and risk managers quantify market risk for the next trading day using historical OHLCV and market cap data. The goal is to achieve low RMSE and MAE on 1-day ahead realized volatility while avoiding time-series data leakage.

## Project Overview

This project implements a complete machine learning pipeline for predicting cryptocurrency volatility from historical price and volume data. It includes:

- **Data Preprocessing**: Handling missing values, data validation, and cleaning
- **Feature Engineering**: Technical indicators, moving averages, Bollinger Bands, liquidity ratios
- **Model Training**: RandomForest regressor for volatility prediction
- **Evaluation**: Comprehensive metrics (RMSE, MAE, R²) on validation and test sets
- **Deployment**: Optional Streamlit app for local predictions

## Repository Structure

```
crypto-volatility-ml/
├── data/
│   └── crypto_prices.csv        # Input dataset (>50 cryptocurrencies)
├── src/
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocess.py            # Data cleaning and splitting
│   ├── features.py              # Feature engineering
│   ├── model.py                 # Model training and evaluation
│   ├── train.py                 # End-to-end training script
│   └── app_streamlit.py         # Streamlit deployment app
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Amannn1/crypto-volatility-ml.git
   cd crypto-volatility-ml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick start

Train the model end-to-end:

```
python -m src.train
```

Run the local Streamlit app:

```
streamlit run src/app_streamlit.py
```

Trained artifacts (model, scaler, feature list) are saved in the `artifacts/` directory and automatically loaded by the Streamlit app.

### Training the Model

Run the full training pipeline:

```bash
python -m src.train
```

This will:
1. Load the crypto price data
2. Clean and preprocess the data
3. Engineer features (moving averages, technical indicators, etc.)
4. Split data into train/val/test sets (time-based)
5. Train a RandomForest model
6. Evaluate on validation and test sets
7. Save artifacts (model, scaler, feature names) to `artifacts/` directory

### Running the Streamlit App

```bash
streamlit run src/app_streamlit.py
```

Then:
1. Upload a CSV file with the same structure as training data
2. View predicted volatility values and visualizations

## Data Format

Expected CSV columns:
- `date` (datetime)
- `symbol` (cryptocurrency symbol, e.g., BTC, ETH)
- `open` (opening price)
- `high` (high price)
- `low` (low price)
- `close` (closing price)
- `volume` (trading volume)
- `market_cap` (market capitalization)

## Features Engineered

- **Returns & Volatility**:
  - Daily returns
  - Rolling volatility (7, 14, 30 days)

- **Technical Indicators**:
  - Moving averages (7, 14, 30 days)
  - Bollinger Bands (20-day middle, upper, lower bands)
  - Average True Range (simple proxy)

- **Liquidity Features**:
  - Liquidity ratio (volume / market_cap)
  - Rolling mean and std of liquidity ratio

- **Calendar Features**:
  - Day of week
  - Month

## Model Performance

### Baselines and model performance

As a simple baseline, the previous day's volatility is used as the prediction. Reported metrics are on the held-out test set using a chronological split to avoid data leakage.

| Model                | RMSE | MAE  | R²   |
|----------------------|------|------|------|
| Naive (prev vol)     | TBD  | TBD  | TBD  |
| RandomForest (ours)  | TBD  | TBD  | TBD  |

*(Fill in actual metrics after training)*

The RandomForest model is evaluated on:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

Metrics are reported on both validation and test sets.

## Key Modules

### `data_loader.py`
Loads CSV data and performs basic datetime conversion and sorting.

### `preprocess.py`
- `clean_data()`: Handles missing values, duplicates, sanity checks
- `scale_features()`: StandardScaler for feature normalization
- `time_based_split()`: Temporal train/val/test split (no data leakage)

### `features.py`
Feature engineering pipeline:
- Returns and rolling volatility
- Liquidity ratios and statistics
- Moving averages and Bollinger Bands
- Calendar features

### `model.py`
- `train_volatility_model()`: Train RandomForest regressor
- `evaluate_model()`: Compute RMSE, MAE, R²

### `train.py`
Orchestrates the full pipeline: load → clean → engineer features → split → scale → train → evaluate → save artifacts

### `app_streamlit.py`
Interactive web app for making predictions on new data.

## Dependencies

See `requirements.txt`:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- joblib

## Future Enhancements

- [High] Integrate live cryptocurrency APIs (CoinGecko / Binance) for near real-time predictions
- [High] Add hyperparameter tuning and cross-validation using `TimeSeriesSplit` for more robust evaluation
- [Medium] Implement LSTM/GRU sequence models as a deep-learning baseline
- [Medium] Add feature importance analysis and visualizations (permutation importance, SHAP)
- [Low] Containerize with Docker and deploy the Streamlit app to Streamlit Community Cloud or Render
- [Low] Build an automated retraining pipeline triggered by new data arrivals
## License

This project is provided as-is for educational purposes.

## Author

Amannn1
