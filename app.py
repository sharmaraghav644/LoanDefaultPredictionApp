import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import bz2

# Function to load non-DL models safely
def load_bz2_model(file_path):
    try:
        with bz2.BZ2File(file_path, "rb") as f:
            return joblib.load(f)  # Use joblib to load compressed models
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load the selected model dynamically
def get_model(choice):
    if choice == "XGBoost":
        return load_bz2_model("xgb_model_compressed.pkl.bz2")
    elif choice == "Random Forest":
        return load_bz2_model("rf_model_compressed.pkl.bz2")
    elif choice == "Deep Learning":
        return load_model("dl_model.keras")  # Corrected to match the .keras file

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
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=1000, max_value=1000000, value=50000)
    loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=500000, value=10000)
    
    education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    marital_status_mapping = {"Single": 0, "Married": 1, "Divorced": 2}
    employment_type_mapping = {"Employed": 0, "Self-Employed": 1, "Unemployed": 2}
    has_co_signer_mapping = {"Yes": 1, "No": 0}
    has_mortgage_mapping = {"Yes": 0, "No": 1}
    has_dependents_mapping = {"Yes": 0, "No": 1}
    loan_purpose_mapping = {
        "Other": 0, "Auto": 1, "Business": 2, "Home": 3, "Education": 4
    }

    education = st.selectbox("Education", options=list(education_mapping.keys()))
    marital_status = st.selectbox("Marital Status", options=list(marital_status_mapping.keys()))
    employment_type = st.selectbox("Employment Type", options=list(employment_type_mapping.keys()))
    has_co_signer = st.selectbox("Has Co-Signer", options=list(has_co_signer_mapping.keys()))
    has_mortgage = st.selectbox("Has Mortgage", options=list(has_mortgage_mapping.keys()))
    has_dependents = st.selectbox("Has Dependents", options=list(has_dependents_mapping.keys()))
    loan_purpose = st.selectbox("Loan Purpose", options=list(loan_purpose_mapping.keys()))

    model_choice = st.selectbox("Select Model for Prediction", ["XGBoost", "Random Forest", "Deep Learning"])

    submitted = st.form_submit_button("Submit")

# If the form is submitted
if submitted:
    # Encode categorical inputs
    education_encoded = education_mapping[education]
    marital_status_encoded = marital_status_mapping[marital_status]
    employment_type_encoded = employment_type_mapping[employment_type]
    has_co_signer_encoded = has_co_signer_mapping[has_co_signer]
    has_mortgage_encoded = has_mortgage_mapping[has_mortgage]
    has_dependents_encoded = has_dependents_mapping[has_dependents]
    loan_purpose_encoded = loan_purpose_mapping[loan_purpose]

    # Default values for missing numerical features
    interest_rate = 0.05
    dti_ratio = 0.2
    credit_score = 650
    num_credit_lines = 5
    loan_term = 30
    months_employed = 24

    # Prepare input DataFrame
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
        "HasMortgage_encoded": [has_mortgage_encoded],
        "HasDependents_encoded": [has_dependents_encoded],
        "LoanPurpose_encoded": [loan_purpose_encoded],
        "HasCoSigner_encoded": [has_co_signer_encoded],
    }

    input_data = pd.DataFrame(input_dict)

    # Apply scaling to user input
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Load the selected model
    st.subheader("Prediction Results")
    with st.spinner("Loading model and predicting..."):
        model = get_model(model_choice)

        if model is None:
            st.error("Failed to load the selected model.")
        else:
            if model_choice in ["XGBoost", "Random Forest"]:
                # Check the prediction probability and print it for debugging
                probability = model.predict_proba(input_data_scaled)[:, 1][0]  # Probability of default
                st.write("Raw Prediction Probability:", probability)  # Debugging step
                percentage = round(probability * 100, 2)

            elif model_choice == "Deep Learning":
                # Get raw output from Deep Learning model
                raw_output = float(model.predict(input_data_scaled)[0][0])  # Ensure scalar output
                st.write("Raw Deep Learning Model Output:", raw_output)  # Debugging step
                percentage = round(raw_output * 100, 2)

            prediction = "Likely to Default" if percentage > 50 else "Not Likely to Default"

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
