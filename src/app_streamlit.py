import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


ARTIFACTS_DIR = Path("artifacts")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "volatility_model.pkl")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    feature_cols = joblib.load(ARTIFACTS_DIR / "feature_cols.pkl")
    return model, scaler, feature_cols


def main():
    st.title("Cryptocurrency Volatility Prediction")

    st.write("Upload a CSV with the same structure as training data (recent days only).")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Raw uploaded data:")
        st.dataframe(df.head())

        model, scaler, feature_cols = load_artifacts()

        # basic feature selection; assumes features already exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing expected feature columns: {missing}")
            return

        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)

        df["predicted_volatility"] = preds
        st.subheader("Predicted volatility")
        st.dataframe(df[["date", "symbol", "predicted_volatility"]])

        st.line_chart(df.set_index("date")["predicted_volatility"])

    st.info("This is a local demo interface as per project guidelines.")
    

if __name__ == "__main__":
    main()
