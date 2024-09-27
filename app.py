import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Load the trained model and scaler with absolute paths
model_path = os.path.join(os.path.dirname(__file__), 'dog_health_model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

# Ensure the model and scaler files exist
if not os.path.exists(model_path):
    st.error("Model file not found. Please ensure the dog_health_model.pkl file is available.")
elif not os.path.exists(scaler_path):
    st.error("Scaler file not found. Please ensure the scaler.pkl file is available.")
else:
    # Load the model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Function to make predictions
    def predict_dog_state(temperature, pulse_rate, heart_rate):
        input_data = pd.DataFrame([[temperature, pulse_rate, heart_rate]],
                                  columns=['temperature', 'pulse_rate', 'heart_rate'])

        # Scale the numerical features
        input_data_scaled = scaler.transform(input_data)

        # Predict state
        state = model.predict(input_data_scaled)
        return state[0]

    # Streamlit app layout
    st.title('Dog Health Analysis')

    st.write('Enter the vital signs of your dog to determine its health status.')

    # Input fields for temperature, pulse rate, and heart rate
    temperature = st.number_input('Temperature (°F)', min_value=90.0, max_value=110.0, value=101.0, step=0.1)
    pulse_rate = st.number_input('Pulse Rate (beats per minute)', min_value=50, max_value=200, value=80)
    heart_rate = st.number_input('Heart Rate (beats per minute)', min_value=50, max_value=200, value=70)

    # Predict button
    if st.button('Predict Health State'):
        try:
            predicted_state = predict_dog_state(temperature, pulse_rate, heart_rate)
            st.write(f'The predicted health state of the dog is: **{predicted_state.capitalize()}**')
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please make sure the model and scaler files are in the same directory as this script.")

    # Display normal ranges
    st.subheader("Normal Ranges for Reference:")
    st.write("Temperature: 99.5°F - 102.5°F")
    st.write("Pulse Rate: 70 - 120 beats per minute")
    st.write("Heart Rate: 60 - 100 beats per minute")
