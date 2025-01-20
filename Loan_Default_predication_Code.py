#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from imblearn.over_sampling import SMOTE
import joblib
import bz2

# Function to save model with compression
def save_model_compressed(model, filename):
    with bz2.BZ2File(filename, 'wb') as f:
        joblib.dump(model, f)

# Data loading and preprocessing code...

file_path = "/Users/raghavsharma/desktop/loan_default_predication_kaggle.csv"

df = pd.read_csv(file_path)

# Names of the columns
print(df.columns)

# Shape of the dataset
print(df.shape)

# First 5 rows
print(df.head(5))

# Data types and missing values
print(df.info())

# Summary statistics
print(df.describe())

missing_percent = df.isnull().sum() / len(df) * 100
print(missing_percent)

# Visualizations...
sns.countplot(x='Default', data=df)
plt.title('Distribution of Loan Default Status')
plt.show()


# Continue with the rest of your preprocessing and feature engineering...
# Example: Encoding categorical columns
label_encoder = LabelEncoder()

# Encode categorical columns that are of object dtype
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

for col in categorical_cols:
    # Apply encoding
    df[col + '_encoded'] = label_encoder.fit_transform(df[col])

df.drop(columns=categorical_cols, inplace=True)
df = df.drop(columns=["LoanID"])

# Verify data types again
print(df.dtypes)

# Data splitting...
X = df.drop('Default', axis=1) # Feature variables
y = df['Default'] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training...
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Making predictions and evaluations...
y_pred_xgb = xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Accuracy: {accuracy_xgb * 100:.2f}%\n")
classification_report_xgb = classification_report(y_test, y_pred_xgb)
print("Classification Report:")
print(classification_report_xgb)

# Using SMOTE to handle class imbalance...
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

xgb_model_smote = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
xgb_model_smote.fit(X_train_smote, y_train_smote)
y_pred_xgb_smote = xgb_model_smote.predict(X_test)

accuracy_xgb_smote = accuracy_score(y_test, y_pred_xgb_smote)
print(f"Accuracy after SMOTE: {accuracy_xgb_smote * 100:.2f}%\n")

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy_rf * 100:.2f}%\n")

# Deep Learning Model
model_dl = Sequential()
model_dl.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model_dl.add(Dense(32, activation='relu'))
model_dl.add(Dense(1, activation='sigmoid'))

# Compile and train the model
model_dl.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_dl.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the models...

# Saving the trained models using compression
save_model_compressed(xgb_model, "xgb_model_compressed.pkl.bz2")
save_model_compressed(rf_model, "rf_model_compressed.pkl.bz2")
save_model_compressed(model_dl, "dl_model_compressed.pkl.bz2")

print("Models have been saved successfully with compression!")

# Optional: Checking the sizes of saved models
import os
xgb_model_size = os.path.getsize("xgb_model_compressed.pkl.bz2") / (1024 * 1024)  # Size in MB
rf_model_size = os.path.getsize("rf_model_compressed.pkl.bz2") / (1024 * 1024)
dl_model_size = os.path.getsize("dl_model_compressed.pkl.bz2") / (1024 * 1024)

print(f"XGB Model Size: {xgb_model_size:.2f} MB")
print(f"Random Forest Model Size: {rf_model_size:.2f} MB")
print(f"Deep Learning Model Size: {dl_model_size:.2f} MB")
