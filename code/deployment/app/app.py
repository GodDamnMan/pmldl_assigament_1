import streamlit as st
import requests
import numpy as np

st.title("ML Model Prediction App")

st.write("Enter feature values for prediction:")

# Example feature input (adjust based on your model)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

if st.button("Predict"):
    features = [feature1, feature2, feature3]
    
    try:
        response = requests.post(
            "http://api:8000/predict",
            json={"features": features}
        )
        
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Prediction: {prediction}")
        else:
            st.error("Error making prediction")
            
    except Exception as e:
        st.error(f"Connection error: {e}")