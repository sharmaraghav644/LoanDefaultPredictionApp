import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import bz2

# Load dataset
file_path = "Loan_Default_predication_kaggle.csv"
df = pd.read_csv(file_path)

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

# Load scaler
scaler = joblib.load("scaler.pkl")  # Make sure you saved the scaler during training

# Streamlit Page Config and Custom Styling
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

# Apply black and white background and text styling
st.markdown(
    """
    <style>
    /* Apply white background to the whole page */
    body {
        background-color: white !important;
        color: black !important;
    }
    /* Sidebar background */
    .sidebar .sidebar-content {
        background-color: white !important;
    }
    /* Streamlit default app background */
    .stApp {
        background-color: white !important;
    }
    /* Button styling - Black background with white text */
    .stButton>button {
        background-color: black !important;
        color: white !important;
    }
    /* Prediction table background and text color */
    .css-1v3fvcr {
        background-color: white !important;
        color: black !important;
    }
    /* Links for repository, portfolio, etc. */
    a {
        color: black !important;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.title("Loan Default Prediction Tool")
st.markdown(
    """
    This app predicts the likelihood of a loan default based on user-provided information.

    Built with **Streamlit**, using **XGBoost** and **Random Forest**.

    For the source code and documentation, visit the repository here - [Github](https://github.com/sharmaraghav644/LoanDefaultPredictionApp)

    Check out my data portfolio - [Data Portfolio](https://raghav-sharma.com/)

    The dataset used in this project is Loan Default Prediction Dataset. You can access it on Kaggle here: [Dataset Link](https://www.kaggle.com/datasets/nikhil1e9/loan-default/data).
    """
)

# Sidebar for input form
st.sidebar.header("Enter Applicant Details")
with st.sidebar.form("user_inputs"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Income", min_value=1000, max_value=1000000, value=50000)
    loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=500000, value=10000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    interest_rate = st.number_input("Interest Rate", min_value=2.0, max_value=25.0, value=10.0, step=0.1)
    months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=60)
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    loan_term = st.number_input("Loan Term (months)", min_value=12, max_value=360, value=36)

    education_mapping = {"Bachelor's": 0, "High School": 1, "Master's": 2, "PhD": 3}
    marital_status_mapping = {"Divorced": 0, "Married": 1, "Single": 2}
    employment_type_mapping = {"Full-time": 0, "Part-time": 1, "Self-employed": 2, "Unemployed": 3}
    has_co_signer_mapping = {"Yes": 1, "No": 0}
    has_mortgage_mapping = {"Yes": 1, "No": 0}
    has_dependents_mapping = {"Yes": 1, "No": 0}
    loan_purpose_mapping = {"Auto": 0, "Business": 1, "Education": 2, "Home": 3, "Other": 4}

    education = st.selectbox("Education", options=list(education_mapping.keys()))
    marital_status = st.selectbox("Marital Status", options=list(marital_status_mapping.keys()))
    employment_type = st.selectbox("Employment Type", options=list(employment_type_mapping.keys()))
    has_co_signer = st.selectbox("Has Co-Signer", options=list(has_co_signer_mapping.keys()))
    has_mortgage = st.selectbox("Has Mortgage", options=list(has_mortgage_mapping.keys()))
    has_dependents = st.selectbox("Has Dependents", options=list(has_dependents_mapping.keys()))
    loan_purpose = st.selectbox("Loan Purpose", options=list(loan_purpose_mapping.keys()))

    model_choice = st.selectbox("Select Model for Prediction", ["XGBoost", "Random Forest"])
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

    # Prepare input DataFrame
    input_dict = {
        "Age": [age],
        "Income": [income],
        "LoanAmount": [loan_amount],
        "CreditScore": [credit_score],
        "MonthsEmployed": [months_employed],
        "NumCreditLines": [int(df["NumCreditLines"].mean())],
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

    # ✅ Apply the scaler to all features at once to match training
    expected_features = scaler.feature_names_in_  # Get feature names from training
    input_data = input_data[expected_features]  # Reorder columns to match training

    # Apply the scaler correctly to all numerical features
    scaled_data = scaler.transform(input_data)
    scaled_input_data = pd.DataFrame(scaled_data, columns=expected_features)

    # Debugging output
    #st.write("Scaled Input Data:")
    #st.write(scaled_input_data)

    # Load the selected model
    st.subheader("Prediction Results")
    with st.spinner("Loading model and predicting..."):
        model = get_model(model_choice)

        if model is None:
            st.error("Failed to load the selected model.")
        else:
            probability = model.predict_proba(scaled_input_data)[:, 1][0]
            percentage = round(probability * 100, 2)
            prediction = "Likely to Default" if probability > 0.5 else "Not Likely to Default"

            st.markdown(
                f"""
                <div style="background-color:white; padding:20px; border-radius:10px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);">
                <h4 style="color:black; font-size:24px; font-weight:600;">Selected Model: {model_choice}</h4> 
                <p style="font-size:18px; color:black; font-weight:400;">{prediction}</p>
                <p style="font-size:18px; color:black; font-weight:400;">Chances of Default: {percentage}%</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
