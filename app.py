import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

st.title("🎓 Student Performance Predictor")

st.write("Enter student details below:")

# User Inputs
study_hours = st.slider("Study Hours per Day", 0, 12, 5)
attendance = st.slider("Attendance (%)", 50, 100, 75)
previous_score = st.slider("Previous Score", 0, 100, 60)

if st.button("Predict"):

    input_data = np.array([[study_hours, attendance, previous_score]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ The student is likely to PASS.")
    else:
        st.error("❌ The student is likely to FAIL.")