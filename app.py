import streamlit as st
import joblib
import numpy as np
import pandas as pd
import bz2
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_url = "https://raw.githubusercontent.com/sharmaraghav644/LoanDefaultPredictionApp/refs/heads/main/Loan_default_predication_kaggle.csv"
df = pd.read_csv(file_url)

# Function to load non-DL models safely
def load_bz2_model(file_path):
    try:
        with bz2.BZ2File(file_path, "rb") as f:
            return joblib.load(f)
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
scaler = joblib.load("scaler.pkl")

# Streamlit Page Config and Custom Styling
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

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
    input_data = pd.DataFrame({
        "Age": [age], "Income": [income], "LoanAmount": [loan_amount], "CreditScore": [credit_score],
        "MonthsEmployed": [months_employed], "NumCreditLines": [int(df["NumCreditLines"].mean())],
        "InterestRate": [interest_rate], "LoanTerm": [loan_term], "DTIRatio": [dti_ratio],
        "Education_encoded": [education_mapping[education]],
        "EmploymentType_encoded": [employment_type_mapping[employment_type]],
        "MaritalStatus_encoded": [marital_status_mapping[marital_status]],
        "HasMortgage_encoded": [has_mortgage_mapping[has_mortgage]],
        "HasDependents_encoded": [has_dependents_mapping[has_dependents]],
        "LoanPurpose_encoded": [loan_purpose_mapping[loan_purpose]],
        "HasCoSigner_encoded": [has_co_signer_mapping[has_co_signer]],
    })
    input_data = input_data[scaler.feature_names_in_]
    scaled_data = scaler.transform(input_data)

    st.subheader("Prediction Results")
    with st.spinner("Loading model and predicting..."):
        model = get_model(model_choice)
        if model:
            probability = model.predict_proba(scaled_data)[:, 1][0]
            st.write(f"Chances of Default: {round(probability * 100, 2)}%")

            st.subheader("Advanced Business Insights")
            if income > 100000 and education in ["Master's", "PhD"]:
                st.write("Targeted Loan Bundles: Consider offering premium loans with lower interest rates for highly qualified, affluent borrowers.")
            if loan_amount > 40000 and income < 40000:
                st.write("Dynamic Loan Amount Caps: High loan amounts in low-income brackets increase default risk. Adjust loan caps accordingly.")
            if loan_amount < 5000:
                st.write("Reevaluate Small Loan Policies: High default rates suggest a need for microfinance coaching or flexible repayment plans.")
            if has_co_signer == "Yes":
                st.write("Incentivize Co-Signed Loans: Co-signed loans reduce default risk. Offering discounts for such cases can be beneficial.")

    st.subheader("Important Features in Loan Default Prediction")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({"Feature": scaler.feature_names_in_, "Importance": importances}).sort_values(by="Importance", ascending=False)
    sns.barplot(x=imp_df["Importance"], y=imp_df["Feature"])
    st.pyplot(plt)
