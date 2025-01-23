import streamlit as st
import joblib
import bz2
import numpy as np
import pandas as pd

# Function to load a compressed model file
def load_compressed_model(model_path):
    with bz2.BZ2File(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

# Load the trained models
xgb_model = load_compressed_model("xgb_model_compressed.pkl.bz2")
rf_model = load_compressed_model("rf_model_compressed.pkl.bz2")
dl_model = load_compressed_model("dl_model_compressed.pkl.bz2")

# Predefined mappings for encoding (must match training data encodings)
education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
marital_status_mapping = {"Single": 0, "Married": 1, "Divorced": 2}
employment_type_mapping = {"Employed": 0, "Self-Employed": 1, "Unemployed": 2}
has_co_signer_mapping = {"Yes": 1, "No": 0}

# Streamlit UI
st.title("Loan Default Prediction")
st.write("Enter the following details to predict loan default:")

# User inputs for features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income", min_value=1000, max_value=1000000, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=500000, value=10000)

# Selecting values for categorical columns
education = st.selectbox("Education", options=list(education_mapping.keys()))
marital_status = st.selectbox("Marital Status", options=list(marital_status_mapping.keys()))
employment_type = st.selectbox("Employment Type", options=list(employment_type_mapping.keys()))
has_co_signer = st.selectbox("Has Co-Signer", options=list(has_co_signer_mapping.keys()))

# Encode the input data based on predefined mappings
education_encoded = education_mapping[education]
marital_status_encoded = marital_status_mapping[marital_status]
employment_type_encoded = employment_type_mapping[employment_type]
has_co_signer_encoded = has_co_signer_mapping[has_co_signer]

# Add missing features with default values
interest_rate = 0.05  # Example default value
dti_ratio = 0.2  # Example default value
credit_score = 650  # Example default value
num_credit_lines = 5  # Example default value
loan_term = 30  # Example default value
months_employed = 24  # Example default value
# Prepare the input as a DataFrame to ensure compatibility with the trained model
# Prepare the input as a DataFrame to ensure compatibility with the trained model
input_dict = {
    "Age": [age],
    "Income": [income],
    "LoanAmount": [loan_amount],
    "CreditScore": [credit_score],
    "MonthsEmployed": [months_employed],
    "NumCreditLines": [num_credit_lines],
    "InterestRate": [interest_rate],
    "LoanTerm": [loan_term],
    "DTIRatio": [dti_ratio],
    "Education_encoded": [education_encoded],
    "MaritalStatus_encoded": [marital_status_encoded],
    "EmploymentType_encoded": [employment_type_encoded],
    "HasCoSigner_encoded": [has_co_signer_encoded],
    "HasMortgage_encoded": [0],  # Default value (adjust as needed)
    "HasDependents_encoded": [0],  # Default value (adjust as needed)
    "LoanPurpose_encoded": [0]  # Default value (adjust as needed)
}

# Create a DataFrame for prediction
input_data = pd.DataFrame(input_dict)

# Ensure feature order matches the trained model
expected_features = xgb_model.feature_names
if list(input_data.columns) != expected_features:
    input_data = input_data[expected_features]

# Prediction
if st.button("Predict"):
    # XGBoost Prediction
    prediction_xgb = xgb_model.predict(input_data)
    if prediction_xgb[0] == 1:
        st.write("XGBoost Model: The loan is likely to default.")
    else:
        st.write("XGBoost Model: The loan is not likely to default.")

    # Random Forest Prediction
    prediction_rf = rf_model.predict(input_data)
    if prediction_rf[0] == 1:
        st.write("Random Forest Model: The loan is likely to default.")
    else:
        st.write("Random Forest Model: The loan is not likely to default.")

    # Deep Learning Model Prediction (threshold for classification)
    prediction_dl_prob = dl_model.predict(input_data)
    prediction_dl = (prediction_dl_prob > 0.5).astype(int)
    if prediction_dl[0] == 1:
        st.write("Deep Learning Model: The loan is likely to default.")
    else:
        st.write("Deep Learning Model: The loan is not likely to default.")