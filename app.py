import streamlit as st
import joblib
import bz2
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

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
    "HasCoSigner_encoded": [has_co_signer_encoded]
}

# Create a DataFrame for prediction
input_data = pd.DataFrame(input_dict)

# Load the dataset to extract y_test for evaluation
file_path = "/Users/raghavsharma/desktop/loan_default_predication_kaggle.csv"
df = pd.read_csv(file_path)

# Encode categorical columns based on your preprocessing steps
label_encoder = LabelEncoder()
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
for col in categorical_cols:
    df[col + '_encoded'] = label_encoder.fit_transform(df[col])
df.drop(columns=categorical_cols, inplace=True)
df = df.drop(columns=["LoanID"])

# Split the data to get y_test for evaluation
X = df.drop('Default', axis=1)
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Prediction and Evaluation
if st.button("Compare Models"):
    # XGBoost Prediction
    y_pred_xgb = xgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    st.write(f"XGBoost Accuracy: {accuracy_xgb * 100:.2f}%")
    st.text(classification_report(y_test, y_pred_xgb))

    # Random Forest Prediction
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    st.write(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
    st.text(classification_report(y_test, y_pred_rf))

    # Deep Learning Model Prediction
    y_pred_dl = (dl_model.predict(X_test) > 0.5).astype(int)
    accuracy_dl = accuracy_score(y_test, y_pred_dl)
    st.write(f"Deep Learning Accuracy: {accuracy_dl * 100:.2f}%")
    st.text(classification_report(y_test, y_pred_dl))
