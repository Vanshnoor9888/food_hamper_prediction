# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1evud8f1zdwBbvruNPUKdX5EJ2dqULSaV
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the SARIMA model
model = joblib.load("sarima_model.pkl")  # Ensure your model file is in the same directory

# Function to make predictions
def predict_hampers(input_date):
    try:
        # Preprocess the input date for SARIMA
        date_index = pd.date_range(start=input_date, periods=1, freq="D")
        # SARIMA expects a time series; here, we pass a dummy series if needed
        dummy_series = pd.Series(np.zeros(len(date_index)), index=date_index)

        # Predict using SARIMA
        prediction = model.forecast(steps=1)[0]
        return round(prediction)
    except Exception as e:
        return f"Error: {e}"

# Streamlit Interface
st.title("Food Hamper Prediction App")
# st.image("download.jpg", caption="Food Drive Project", use_column_width=True)

st.markdown("""
### 📦 Predict the Number of Food Hampers
Enter a date to predict the number of hampers that may be needed.
""")

# User input
input_date = st.date_input("Select a Date:", value=pd.Timestamp.now().date())

# Predict button
if st.button("Predict"):
    result = predict_hampers(input_date)
    st.success(f"The predicted number of food hampers for {input_date} is: {result}")

st.markdown("""
---
📍 **Note:** This model uses historical data and SARIMA for predictions. Ensure that the model is trained properly for accuracy.
""")