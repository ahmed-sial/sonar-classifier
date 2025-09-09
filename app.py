import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load the trained pipeline
pipeline = joblib.load("sonar-model.pkl")

# Streamlit UI
st.set_page_config(page_title="Sonar Classification", page_icon="🌊", layout="wide")

st.title("🌊 Sonar Dataset Classifier")
st.write("Upload a CSV file with sonar data (60 features per row and 1 optional label column).")

# File uploader
uploaded_file = st.file_uploader("Upload your sonar CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file, header=None)
    st.subheader("📊 Uploaded Data")
    st.dataframe(df)

    # If labels are present (61 columns), separate them
    if df.shape[1] == 61:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        st.success("✅ Detected labels in uploaded file")
    else:
        X = df  # assume only features
        y = None
        st.info("No labels detected (only features).")

    # Predict button
    if st.button("⚡ Predict with Sonar Model"):
        with st.spinner("Running predictions... ⏳"):
            time.sleep(1)
            try:
                preds = pipeline.predict(X)
                preds = np.array(preds)  # ensure numpy array
                st.subheader("🔮 Predictions")
                df_result = df.copy()
                df_result["Prediction"] = np.where(preds == 0, "M", "R")
                st.dataframe(df_result)
                csv = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Predictions as CSV",
                    data=csv,
                    file_name="sonar_prediction.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}\n\nMake sure your file has 60 features (and optionally 1 label column).")