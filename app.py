import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load('xgboost_creditcardfraud_model.pkl') 

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")

st.markdown("Enter the transaction features below to predict if it's **fraudulent** or **legitimate**.")

# Input fields for V1 to V28, Time and Amount
feature_names = [f'V{i}' for i in range(1, 29)] + ['Time'] + ['Amount']
user_input = []

# Organize in 3 columns for better layout
cols = st.columns(3)

for i, feature in enumerate(feature_names):
    value = cols[i % 3].number_input(f'{feature}', value=0.0, format="%.5f")
    user_input.append(value)

# Predict button
if st.button("🔍 Predict"):
    input_array = np.array([user_input])
    prediction = model.predict(input_array)[0]
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")