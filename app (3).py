import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import inv_boxcox
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load your preprocessed data and trained SARIMA model
@st.cache_data
def load_data_and_model():
    # Load your dataset (replace 'your_dataset.csv')
    df = pd.read_csv('dataframe.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Define your SARIMA model parameters
    sarima_model = sm.tsa.SARIMAX(
        df['actual_pickup_boxcox'],
        exog=df[['scheduled_pickup', 'scheduled_pickup_lag_7', 'scheduled_pickup_lag_14']],
        order=(4, 1, 4),
        seasonal_order=(1, 1, 1, 7)
    )

    sarima_fit = sarima_model.fit(disp=False)
    return df, sarima_fit

df, sarima_fit = load_data_and_model()

# Page Title and Instructions
st.title("Food Hamper Pickup Predictor")
st.write("Enter the date range to predict food hamper pickups and visualize the results.")

# User Input: Start and End Dates
start_date = st.text_input("Start Date (YYYY-MM-DD):", "2024-01-01")
end_date = st.text_input("End Date (YYYY-MM-DD):", "2024-01-15")

if st.button("Predict"):
    try:
        # Convert input dates to datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Generate prediction dates
        prediction_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Prepare exogenous variables for prediction
        exog_future = df.loc[df['date'].isin(prediction_dates), 
                             ['scheduled_pickup', 'scheduled_pickup_lag_7', 'scheduled_pickup_lag_14']]
        if exog_future.empty:
            st.error("No data available for the specified dates. Please choose another range.")
        else:
            # Forecast with SARIMA model
            forecast = sarima_fit.get_forecast(steps=len(exog_future), exog=exog_future)
            forecast_values_boxcox = forecast.predicted_mean

            # Reverse Box-Cox Transformation
            forecast_values_original = inv_boxcox(forecast_values_boxcox, sarima_fit.params['lambda'])

            # Create a DataFrame for predictions
            prediction_df = pd.DataFrame({
                'date': prediction_dates,
                'predicted_pickups': forecast_values_original
            })

            # Display predictions
            st.subheader("Predicted Food Hamper Pickups")
            st.dataframe(prediction_df)

            # Plot predictions
            st.subheader("Prediction Graph")
            plt.figure(figsize=(10, 6))
            plt.plot(prediction_df['date'], prediction_df['predicted_pickups'], marker='o', label='Predicted Pickups')
            plt.title('Predicted Food Hamper Pickups')
            plt.xlabel('Date')
            plt.ylabel('Number of Pickups')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"An error occurred: {e}")
