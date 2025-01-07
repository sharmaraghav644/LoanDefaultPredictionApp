import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained models
xgb_model = joblib.load("xgb_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# Function to encode the input data
def encode_input_data(education, marital_status, employment_type, has_co_signer):
    # These values should match the encoding you performed during training
    label_encoder = LabelEncoder()
    
    # Encode user input values (you can use the exact encoder values or redefine a new LabelEncoder)
    education_encoded = label_encoder.fit_transform([education])[0]
    marital_status_encoded = label_encoder.fit_transform([marital_status])[0]
    employment_type_encoded = label_encoder.fit_transform([employment_type])[0]
    has_co_signer_encoded = label_encoder.fit_transform([has_co_signer])[0]
    
    return education_encoded, marital_status_encoded, employment_type_encoded, has_co_signer_encoded

# Streamlit UI
st.title("Loan Default Prediction")
st.write("Enter the following details to predict loan default:")

# User inputs for features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=1000, max_value=1000000, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=500000, value=10000)

# Selecting values for categorical columns
education = st.selectbox("Education", options=["High School", "Bachelor's", "Master's", "PhD"])
marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
employment_type = st.selectbox("Employment Type", options=["Employed", "Self-Employed", "Unemployed"])
has_co_signer = st.selectbox("Has Co-Signer", options=["Yes", "No"])

# Encode the input data
education_encoded, marital_status_encoded, employment_type_encoded, has_co_signer_encoded = encode_input_data(
    education, marital_status, employment_type, has_co_signer)

# Feature array for prediction
input_data = np.array([age, income, loan_amount, education_encoded, marital_status_encoded, employment_type_encoded, has_co_signer_encoded]).reshape(1, -1)

# Prediction
if st.button("Predict"):
    prediction_xgb = xgb_model.predict(input_data)
    prediction_rf = rf_model.predict(input_data)

    if prediction_xgb[0] == 1:
        st.write("XGBoost Model: The loan is likely to default.")
    else:
        st.write("XGBoost Model: The loan is not likely to default.")

    if prediction_rf[0] == 1:
        st.write("Random Forest Model: The loan is likely to default.")
    else:
        st.write("Random Forest Model: The loan is not likely to default.")
