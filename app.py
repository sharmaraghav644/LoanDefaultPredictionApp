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

# Predefined mappings for encoding (must match training data encodings)
education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
marital_status_mapping = {"Single": 0, "Married": 1, "Divorced": 2}
employment_type_mapping = {"Employed": 0, "Self-Employed": 1, "Unemployed": 2}
has_co_signer_mapping = {"Yes": 1, "No": 0}

# Streamlit Page Config and Custom Styling
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

# Header Section
st.title("Loan Default Prediction Tool")
st.markdown(
    """
    This app predicts the likelihood of a loan default based on user-provided information.

    Built with **Streamlit**, using **XGBoost**, **Random Forest**, and a **Deep Learning** model.

    The dataset used in this project is Loan Default Prediction Dataset. You can access it on Kaggle here: [Dataset Link](https://www.kaggle.com/datasets/nikhil1e9/loan-default/data).
    """
)

# Sidebar for input form
st.sidebar.header("Enter Applicant Details")
with st.sidebar.form("user_inputs"):
    # Input fields
    age = st.number_input(
        "Age", min_value=18, max_value=100, value=30, help="Applicant's age (in years)."
    )
    income = st.number_input(
        "Income", min_value=1000, max_value=1000000, value=50000, help="Annual income (in USD)."
    )
    loan_amount = st.number_input(
        "Loan Amount", min_value=1000, max_value=500000, value=10000, help="Requested loan amount (in USD)."
    )
    education = st.selectbox(
        "Education", options=list(education_mapping.keys()), help="Highest level of education completed."
    )
    marital_status = st.selectbox(
        "Marital Status", options=list(marital_status_mapping.keys()), help="Applicant's marital status."
    )
    employment_type = st.selectbox(
        "Employment Type",
        options=list(employment_type_mapping.keys()),
        help="Current employment status of the applicant.",
    )
    has_co_signer = st.selectbox(
        "Has Co-Signer",
        options=list(has_co_signer_mapping.keys()),
        help="Whether the applicant has a co-signer for the loan.",
    )
    
    # Dropdown to select model
    model_choice = st.selectbox(
        "Select Model for Prediction", 
        ["XGBoost", "Random Forest", "Deep Learning"],
        help="Choose which model to use for loan default prediction."
    )
    
    submitted = st.form_submit_button("Submit")

# Function to load the selected model dynamically
def get_model(choice):
    if choice == "XGBoost":
        return load_compressed_model("xgb_model_compressed.pkl.bz2")
    elif choice == "Random Forest":
        return load_compressed_model("rf_model_compressed.pkl.bz2")
    elif choice == "Deep Learning":
        return load_compressed_model("dl_model_compressed.pkl.bz2")

# If the form is submitted
if submitted:
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
        "EmploymentType_encoded": [employment_type_encoded],
        "MaritalStatus_encoded": [marital_status_encoded],
        "HasMortgage_encoded": [0],  # Default value (adjust as needed)
        "HasDependents_encoded": [0],  # Default value (adjust as needed)
        "LoanPurpose_encoded": [0],  # Default value (adjust as needed)
        "HasCoSigner_encoded": [has_co_signer_encoded],
    }

    input_data = pd.DataFrame(input_dict)

    # Main Area for Results
    st.subheader("Prediction Results")
    with st.spinner("Loading model and predicting..."):
        model = get_model(model_choice)
        if model_choice in ["XGBoost", "Random Forest"]:
            probability = model.predict_proba(input_data)[0][1]  # Probability of default
        elif model_choice == "Deep Learning":
            probability = model.predict(input_data)[0][0]  # Deep learning may output probability directly

        prediction = "Likely to Default" if probability > 0.5 else "Not Likely to Default"
        percentage = round(probability * 100, 2)

    # Display Prediction and Percentage
    st.markdown(
        f"""
        <div style="background-color:#ffffff; padding:20px; border-radius:10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);">
        <h4 style="color:#2c3e50; font-size:24px; font-weight:600;">Selected Model: {model_choice}</h4> 
        <p style="font-size:18px; color:#2c3e50; font-weight:400;">{prediction}</p>
        <p style="font-size:18px; color:#2c3e50; font-weight:400;">Chances of Default: {percentage}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Clear Cache after Prediction
st.cache_data.clear()