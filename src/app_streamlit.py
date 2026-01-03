import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ARTIFACTS_DIR = Path("artifacts")


@st.cache_resource
def load_artifacts():
    """Load trained model and preprocessing artifacts."""
    model = joblib.load(ARTIFACTS_DIR / "volatility_model.pkl")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    feature_cols = joblib.load(ARTIFACTS_DIR / "feature_cols.pkl")
    return model, scaler, feature_cols


def validate_csv_schema(df, required_cols):
    """Validate that CSV has all required columns."""
    missing = [c for c in required_cols if c not in df.columns]
    return missing


def main():
    st.set_page_config(page_title="Volatility Prediction", layout="wide")
    st.title("Cryptocurrency Volatility Prediction")
    st.markdown("Predict 1-day ahead volatility using RandomForest on historical OHLCV and market cap data.")
    
    # Sidebar: instructions and data format
    with st.sidebar:
        st.markdown("### Upload Instructions")
        st.markdown("""
        1. **Prepare CSV** with these columns:
           - `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`, `market_cap`
           - Columns can also include pre-engineered features from training
        2. **Upload File** and view predictions below
        3. (Optional) If `target_volatility` column is present, metrics will be shown
        """)
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        base_cols = ["date", "symbol", "open", "high", "low", "close", "volume", "market_cap"]
        missing = validate_csv_schema(df, base_cols)
        
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
            st.info(f"Expected columns: {base_cols}")
            return
        
        # Load artifacts
        model, scaler, feature_cols = load_artifacts()
        
        # Validate that all feature columns exist
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            st.error(f"❌ Missing feature columns (likely need preprocessing): {missing_features}")
            return
        
        # Extract features and make predictions
        X = df[feature_cols].copy()
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        
        # Add predictions to dataframe
        df["predicted_volatility"] = preds
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["Raw Data", "Predictions", "Visualizations"])
        
        with tab1:
            st.subheader("Uploaded Data (first 10 rows)")
            st.dataframe(df[base_cols].head(10))
        
        with tab2:
            st.subheader("Predictions")
            result_cols = ["date", "symbol", "predicted_volatility"]
            if "target_volatility" in df.columns:
                result_cols.insert(2, "target_volatility")
            st.dataframe(df[result_cols], use_container_width=True)
        
        with tab3:
            st.subheader("Volatility Predictions Over Time")
            
            if "date" in df.columns:
                df_plot = df[["date", "predicted_volatility"]].copy()
                df_plot["date"] = pd.to_datetime(df_plot["date"], errors='coerce')
                df_plot = df_plot.dropna()
                
                if len(df_plot) > 0:
                    chart_df = df_plot.set_index("date")
                    st.line_chart(chart_df)
                else:
                    st.warning("Could not parse dates for visualization.")
            else:
                st.line_chart(df[["predicted_volatility"]])
        
        # Display metrics if target is available
        if "target_volatility" in df.columns:
            st.subheader("Model Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            
            rmse = np.sqrt(mean_squared_error(df["target_volatility"], df["predicted_volatility"]))
            mae = mean_absolute_error(df["target_volatility"], df["predicted_volatility"])
            r2 = r2_score(df["target_volatility"], df["predicted_volatility"])
            
            with col1:
                st.metric("RMSE", f"{rmse:.4f}")
            with col2:
                st.metric("MAE", f"{mae:.4f}")
            with col3:
                st.metric("R² Score", f"{r2:.3f}")
            
            # Actual vs Predicted plot
            st.subheader("Actual vs Predicted Volatility")
            comparison_df = pd.DataFrame({
                "Actual": df["target_volatility"],
                "Predicted": df["predicted_volatility"]
            })
            st.line_chart(comparison_df)
    
    else:
        st.info("👆 Please upload a CSV file to get started.")


if __name__ == "__main__":
    main()
