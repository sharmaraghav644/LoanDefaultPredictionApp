import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
file_path = "/Users/raghavsharma/desktop/loan_default_predication_kaggle.csv"
df = pd.read_csv(file_path)

# Print the column names
#print(df.columns)

# Print the shape of the dataset
#print(f"Shape of the dataset: {df.shape}")

# Display first 5 rows
#print(df.head(5))

# Check for missing values in the dataset
#missing_percent = df.isnull().sum() / len(df) * 100
#print("Missing value percentage:\n", missing_percent)

# Drop LoanID if it exists (ensure the column name is correct)
if 'LoanID' in df.columns:
    df.drop(columns=['LoanID'], inplace=True)

# Handle missing values (optional, can also fill or drop missing values as needed)
df = df.dropna()  # Dropping rows with missing values, or you can choose to fill with df.fillna()

# Encoding categorical columns using LabelEncoder
label_encoder = LabelEncoder()

# List of categorical columns
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

# Encode each categorical column and drop the original columns
for col in categorical_cols:
    if col in df.columns:
        df[col + '_encoded'] = label_encoder.fit_transform(df[col])
        df.drop(columns=[col], inplace=True)

# Apply numerical conversion to the dataframe
df = df.apply(pd.to_numeric, errors='coerce')

# Splitting the data into features and target variable
X = df.drop('Default', axis=1)  # Features
y = df['Default']  # Target Variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBClassifier model
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)

# Training the model
xgb_model.fit(X_train, y_train)

# Making predictions
y_pred_xgb = xgb_model.predict(X_test)

#Evaluation

# Evaluating the model
accuracy_xgb = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_xgb * 100:.2f}%\n")

classification_report_xgb = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report_xgb)

confusion_matrix_xgb = confusion_matrix(y_test, y_pred)
# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix_xgb)

# Initialize the RandomForestClassifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy_rf * 100:.2f}%\n")

classification_report_rf = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_report_rf)

confusion_matrix_rf = confusion_matrix(y_test, y_pred)
# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix_rf)

# Save the trained models as .pkl files
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")

print("Models have been saved successfully!")

# Create a DataFrame for comparison
comparison_table = pd.DataFrame({
    "Metric": ["Accuracy", "Classification Report", "Confusion Matrix"],
    "XGB Model": [accuracy_xgb, classification_report_xgb, confusion_matrix_xgb],
    "Random Forest Model": [accuracy_rf, classification_report_rf, confusion_matrix_rf]
})

# Format the table using tabulate
print(tabulate(comparison_table, headers="keys", tablefmt="pretty")
