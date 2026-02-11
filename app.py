import sklearn
import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Fall Detection", layout="centered")

# Title
st.title("🚨 Fall Detection Prediction")
st.write("Enter patient vitals and sensor data to predict fall risk")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('knn_best_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'knn_best_model.pkl' not found. Please ensure the model is saved.")
        return None

model = load_model()

# Create input fields as a dictionary
st.subheader("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    distance = st.number_input('Distance (cm)', min_value=0.0, max_value=70.0, value=3.585)

with col2:
    pressure = st.number_input('Pressure (mmHg)', min_value=0.0, max_value=2.0, value=2.0)

with col3:
    hrv = st.number_input('Heart Rate Variability (ms)', min_value=0.0, max_value=125.0, value=112.170)

col4, col5, col6 = st.columns(3)

with col4:
    sugar = st.number_input('Sugar Level (mg/dL)', min_value=10.0, max_value=180.0, value=24.0)

with col5:
    spo2 = st.number_input('SpO2 (%)', min_value=60.0, max_value=100.0, value=67.0)

with col6:
    accelerometer = st.number_input('Accelerometer (m/s²)', min_value=0.0, max_value=1.0, value=1.0)

features_dict = {
    'Distance': distance,
    'Pressure': pressure,
    'HRV': hrv,
    'Sugar level': sugar,
    'SpO2': spo2,
    'Accelerometer': accelerometer,
}

# Display the input dictionary
# st.subheader("Input Data Dictionary")
# st.json(features_dict)

# Make prediction
if st.button("Predict Fall Risk", type="primary"):
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
    else:
        try:
            # Prepare input data
            feature_names = ['Distance', 'Pressure', 'HRV', 'Sugar level', 'SpO2', 'Accelerometer']
            input_data = np.array([features_dict[feature] for feature in feature_names]).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            if prediction == 2:
                st.error("⚠️ HIGH RISK OF FALL DETECTED")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Level", "HIGH", delta=f"{probability[1]*100:.1f}%")
                with col2:
                    st.metric("Confidence", f"{max(probability)*100:.1f}%")
            elif prediction == 1:
                st.error("⚠️ PROBABLE RISK OF FALL DETECTED")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Level", "HIGH", delta=f"{probability[1]*100:.1f}%")
                with col2:
                    st.metric("Confidence", f"{max(probability)*100:.1f}%")
            else:
                st.success("✅ LOW RISK - No Fall Detected")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Risk Level", "LOW", delta=f"{probability[0]*100:.1f}%")
                with col2:
                    st.metric("Confidence", f"{max(probability)*100:.1f}%")
            
            # Show probability breakdown
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Class': ['No Fall (0)', 'Fall Risk (1)'],
                'Probability': probability
            })
            st.bar_chart(prob_df.set_index('Class'))
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Footer
st.divider()
st.caption("Fall Detection System - Powered by Scikit-learn")
data = {'Distance': 'Distance', 'Pressure': 'Pressure', 'HRV': 'HRV', 'Sugar level': 'Sugar level', 'SpO2': 'SpO2', 'Accelerometer': 'Accelerometer', 'Decision': 'Decision'}