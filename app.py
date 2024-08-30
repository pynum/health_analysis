def predict_dog_state(temperature, pulse_rate, heart_rate):
    log_message(f"Prediction requested for Temperature: {temperature}, Pulse Rate: {pulse_rate}, Heart Rate: {heart_rate}")
    
    if model is None or scaler is None:
        log_message("Error: Model or scaler not loaded")
        return "Error: Model or scaler not loaded"
    
    input_data = np.array([[temperature, pulse_rate, heart_rate]])
    log_message(f"Input data shape: {input_data.shape}")
    
    try:
        input_data_scaled = scaler.transform(input_data)
        log_message(f"Scaled input data shape: {input_data_scaled.shape}")
    except Exception as e:
        log_message(f"Error during scaling: {str(e)}")
        return f"Error: Unable to scale input data - {str(e)}"

    try:
        state = model.predict(input_data_scaled)
        log_message(f"Predicted state: {state[0]}")
        return state[0]
    except Exception as e:
        log_message(f"Error during prediction: {str(e)}")
        return f"Error: Unable to make prediction - {str(e)}"
