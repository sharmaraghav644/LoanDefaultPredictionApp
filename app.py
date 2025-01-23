import streamlit as st
import joblib
import bz2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.metrics import accuracy_score
import shap

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

# Assuming your test data is stored as a variable named 'X_test' and 'y_test'
# Example: You need to load your test data here. If it's from a CSV file:
# X_test = pd.read_csv("test_data.csv")
# y_test = X_test['default_label_column']

# Prediction
if st.button("Predict"):
    # Predictions for all models
    prediction_xgb = xgb_model.predict(input_data)
    prediction_rf = rf_model.predict(input_data)
    prediction_dl_prob = dl_model.predict(input_data)
    prediction_dl = (prediction_dl_prob > 0.5).astype(int)

    # Show model predictions
    st.write(f"XGBoost Model: {'The loan is likely to default' if prediction_xgb[0] == 1 else 'The loan is not likely to default'}")
    st.write(f"Random Forest Model: {'The loan is likely to default' if prediction_rf[0] == 1 else 'The loan is not likely to default'}")
    st.write(f"Deep Learning Model: {'The loan is likely to default' if prediction_dl[0] == 1 else 'The loan is not likely to default'}")

    # Model comparison with metrics
    st.subheader("Model Comparison Metrics")

    # Evaluate XGBoost Model
    y_pred_xgb = xgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

    # Evaluate Random Forest Model
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    # Evaluate Deep Learning Model
    y_pred_dl = (dl_model.predict(X_test) > 0.5).astype(int)
    accuracy_dl = accuracy_score(y_test, y_pred_dl)

    # Display metrics
    st.write(f"XGBoost Accuracy: {accuracy_xgb * 100:.2f}%")
    st.write(f"Random Forest Accuracy: {accuracy_rf * 100:.2f}%")
    st.write(f"Deep Learning Accuracy: {accuracy_dl * 100:.2f}%")

    # Classification report for each model
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred_xgb))
    st.text(classification_report(y_test, y_pred_rf))
    st.text(classification_report(y_test, y_pred_dl))

    # Confusion Matrix for each model
    st.subheader("Confusion Matrix")
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_dl = confusion_matrix(y_test, y_pred_dl)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues", ax=ax[0], cbar=False)
    ax[0].set_title("XGBoost Confusion Matrix")
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=ax[1], cbar=False)
    ax[1].set_title("Random Forest Confusion Matrix")
    sns.heatmap(cm_dl, annot=True, fmt="d", cmap="Blues", ax=ax[2], cbar=False)
    ax[2].set_title("Deep Learning Confusion Matrix")
    st.pyplot(fig)

    # ROC Curves
    st.subheader("ROC Curves")
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_model.predict_proba(X_test)[:,1])
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
    fpr_dl, tpr_dl, _ = roc_curve(y_test, dl_model.predict_proba(X_test)[:,1])

    auc_xgb = auc(fpr_xgb, tpr_xgb)
    auc_rf = auc(fpr_rf, tpr_rf)
    auc_dl = auc(fpr_dl, tpr_dl)

    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr_xgb, tpr_xgb, color='blue', label=f'XGBoost (AUC = {auc_xgb:.2f})')
    ax_roc.plot(fpr_rf, tpr_rf, color='green', label=f'Random Forest (AUC = {auc_rf:.2f})')
    ax_roc.plot(fpr_dl, tpr_dl, color='red', label=f'Deep Learning (AUC = {auc_dl:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax_roc.set_title('ROC Curves for Each Model')
    st.pyplot(fig_roc)
