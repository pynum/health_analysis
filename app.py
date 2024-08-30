import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

# Attempt to import joblib, with a fallback to pickle
try:
    import joblib
except ImportError:
    import pickle as joblib
    st.warning("Using pickle instead of joblib. Consider installing joblib for better performance.")

st.write(f"scikit-learn version: {sklearn.__version__}")

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('dog_health_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure the files are in the correct location.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

model, scaler = load_model_and_scaler()

# Function to make predictions
def predict_dog_state(temperature, pulse_rate, heart_rate):
    if model is None or scaler is None:
        return "Error: Model or scaler not loaded"
    
    input_data = pd.DataFrame([[temperature, pulse_rate, heart_rate]],
                              columns=['temperature', 'pulse_rate', 'heart_rate'])

    # Scale the numerical features
    input_data_scaled = scaler.transform(input_data)

    # Predict state
    try:
        state = model.predict(input_data_scaled)
        return state[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return "Error: Unable to make prediction"

# Streamlit app layout
st.title('Dog Health Analysis')

st.write('Enter the vital signs of your dog to determine its health status.')

# Input fields for temperature, pulse rate, and heart rate
temperature = st.number_input('Temperature (°F)', min_value=90.0, max_value=110.0, value=101.0, step=0.1)
pulse_rate = st.number_input('Pulse Rate (beats per minute)', min_value=50, max_value=200, value=80)
heart_rate = st.number_input('Heart Rate (beats per minute)', min_value=50, max_value=200, value=70)

# Predict button
if st.button('Predict Health State'):
    if model is not None and scaler is not None:
        predicted_state = predict_dog_state(temperature, pulse_rate, heart_rate)
        if not predicted_state.startswith("Error"):
            st.write(f'The predicted health state of the dog is: **{predicted_state.capitalize()}**')
    else:
        st.error("Cannot make predictions. Model or scaler not loaded.")

# Display normal ranges
st.subheader("Normal Ranges for Reference:")
st.write("Temperature: 99.5°F - 102.5°F")
st.write("Pulse Rate: 70 - 120 beats per minute")
st.write("Heart Rate: 60 - 100 beats per minute")
