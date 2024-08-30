import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Dog Health Analysis", page_icon="🐶", layout="wide")

# Logging function
def log_message(message):
    st.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

log_message("App started")

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    log_message("Attempting to load model and scaler")
    try:
        start_time = time.time()
        with open('dog_health_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler_data = pickle.load(f)
        
        # Check if scaler_data is a numpy array
        if isinstance(scaler_data, np.ndarray):
            log_message("Scaler data is a numpy array. Creating StandardScaler.")
            scaler = StandardScaler()
            scaler.mean_ = scaler_data[0]
            scaler.scale_ = scaler_data[1]
        else:
            scaler = scaler_data
        
        load_time = time.time() - start_time
        log_message(f"Model and scaler loaded successfully in {load_time:.2f} seconds")
        return model, scaler
    except FileNotFoundError as e:
        log_message(f"File not found error: {str(e)}")
        st.error("Model or scaler file not found. Please ensure the files are in the correct location.")
        return None, None
    except Exception as e:
        log_message(f"Error loading model or scaler: {str(e)}")
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

log_message("Calling load_model_and_scaler()")
model, scaler = load_model_and_scaler()
log_message(f"Finished calling load_model_and_scaler(). Model type: {type(model)}, Scaler type: {type(scaler)}")

# Function to make predictions
def predict_dog_state(temperature, pulse_rate, heart_rate):
    if model is None or scaler is None:
        return "Error: Model or scaler not loaded"
    
    input_data = np.array([[temperature, pulse_rate, heart_rate]])
    
    try:
        input_data_scaled = scaler.transform(input_data)
    except Exception as e:
        log_message(f"Error during scaling: {str(e)}")
        return f"Error: Unable to scale input data - {str(e)}"

    try:
        state = model.predict(input_data_scaled)
        return state[0]
    except Exception as e:
        log_message(f"Error during prediction: {str(e)}")
        return f"Error: Unable to make prediction - {str(e)}"

# Streamlit app layout
st.title('Dog Health Analysis')

st.write('Enter the vital signs of your dog to determine its health status.')

# Input fields for temperature, pulse rate, and heart rate
temperature = st.number_input('Temperature (°F)', min_value=90.0, max_value=110.0, value=101.0, step=0.1)
pulse_rate = st.number_input('Pulse Rate (beats per minute)', min_value=50, max_value=200, value=80)
heart_rate = st.number_input('Heart Rate (beats per minute)', min_value=50, max_value=200, value=70)

# Predict button
if st.button('Predict Health State'):
    log_message("Prediction button clicked")
    if model is not None and scaler is not None:
        predicted_state = predict_dog_state(temperature, pulse_rate, heart_rate)
        if not predicted_state.startswith("Error"):
            st.write(f'The predicted health state of the dog is: **{predicted_state.capitalize()}**')
        else:
            st.error(predicted_state)
    else:
        st.error("Cannot make predictions. Model or scaler not loaded.")

# Display normal ranges
st.subheader("Normal Ranges for Reference:")
st.write("Temperature: 99.5°F - 102.5°F")
st.write("Pulse Rate: 70 - 120 beats per minute")
st.write("Heart Rate: 60 - 100 beats per minute")

# Display environment information
st.subheader("Debug Information:")
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Files in current directory: {os.listdir('.')}")
log_message("App finished loading")
