

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


file_path = "/Users/raghavsharma/desktop/loan_default_predication_kaggle.csv"

df = pd.read_csv(file_path)
#Names of the columns
print(df.columns)
# Shape of the dataset
print(df.shape)

# First 5 rows
print(df.head(5))


# Data types and missing values
print(df.info())

# Summary statistics
print(df.describe())
print(df.isnull().sum)
missing_percent = df.isnull().sum() / len(df) * 100
print(missing_percent)

#Distribution of the Target Variable (Default)

#Why do we do this? We want to know if most of the loans are "good" loans (not defaulted) or "bad" loans (defaulted). 
# If we have way more loans that didn’t default, the model might get biased towards predicting "no default" because that’s what it’s mostly seeing.

sns.countplot(x='Default', data=df)
plt.title('Distribution of Loan Default Status')
plt.show()

default_grouped = df.groupby('Default').size()

#0 indicates approved
#1 indicated not approved

# Print the result
print(default_grouped)
#Univariate Analysis of Continuous Features


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for Applicant Income Distribution
sns.histplot(df['Income'], kde=True, ax=axes[0], color='blue')
axes[0].set_title('Income Distribution')
axes[0].set_xlabel('Income')
axes[0].set_ylabel('Frequency')

# Plot for Loan Amount Distribution
sns.histplot(df['LoanAmount'], kde=True, ax=axes[1], color='green')
axes[1].set_title('Loan Amount Distribution')
axes[1].set_xlabel('Loan Amount')
axes[1].set_ylabel('Frequency')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

sns.boxplot(x='Default', y='Income', data=df)
plt.title('Applicant Income by Loan Default Status')
plt.show()

#Analysis of the Categorical Categories

sns.countplot(x='MaritalStatus', data=df)
plt.title('Marital Status Distribution')
plt.show()

sns.countplot(x='Education', data=df)
plt.title('Education Distribution')
plt.show()
#Correlation between features

corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
sns.pairplot(df[['Income', 'LoanAmount', 'InterestRate', 'Default']], hue='Default')
plt.title('Pairwise Relationships')
plt.show()
#In the dataset, the column LoanID is a unique identifier for each loan application and does not carry any meaningful information that could help the model 
# predict whether a loan will default or not.

#The correct course of action is to drop the column before the processing of the data.

df.drop(columns=['LoanID'], inplace=True)

print(df.head(5))
#A few of the columns (Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCosigner) have categorical values. 
# Before we proceed, we need to convert them into numerical values.

label_encoder = LabelEncoder()

# Apply Label Encoding to the 'purpose' column since it is the only categorical value column
df['EmploymentType_encoded'] = label_encoder.fit_transform(df['EmploymentType'])
df['MarritalStatus_encoded'] = label_encoder.fit_transform(df['MaritalStatus'])
df['Education_encoded'] = label_encoder.fit_transform(df['Education'])
df['HasMortgage_encoded'] = label_encoder.fit_transform(df['HasMortgage'])
df['HasDependents_encoded'] = label_encoder.fit_transform(df['HasDependents'])
df['LoanPurpose_encoded'] = label_encoder.fit_transform(df['LoanPurpose'])
df['HasCoSigner_encoded'] = label_encoder.fit_transform(df['HasCoSigner'])

print(df.columns)
print(df.head(4))
# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical columns that are of object dtype
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

for col in categorical_cols:
    # Apply encoding
    df[col + '_encoded'] = label_encoder.fit_transform(df[col])

# Drop original categorical columns after encoding
df.drop(columns=categorical_cols, inplace=True)

# Verify data types again
print(df.dtypes)
df = df.apply(pd.to_numeric, errors='coerce')

# Verify data types again
print(df.dtypes)
Processing and modelling the data
X = df.drop('Default', axis=1) #feature variables
y = df['Default'] #Target Variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#X_train: The features or input data the model learns from during training.
#Y_train: The correct outcomes or labels corresponding to X_train that the model aims to predict.
#X_test: The new, unseen data the model uses to make predictions after training.
#Y_test: The actual correct outcomes for the test data, used to evaluate the model's performance.
#y_pred: The predictions made by the model for the test data, based on what it learned.


We will be utilizing three different algorithms to analyzw this data - XgBoost, RandomForest and neural networks using deep learning.
XG Boost (Extreme Gradient Boosting)

Why it's used:

High Predictive Power: XGBoost is one of the most powerful algorithms for structured/tabular data. It is designed to optimize performance and handles complex relationships well.
Handles Imbalanced Data: Loan default datasets are often imbalanced (e.g., more loans are approved than defaulted). XGBoost uses techniques like weighted loss functions to deal with this.


Feature Importance: It provides insights into which features are most important for predicting defaults, which can help with feature selection and model interpretability.
Handles Missing Data: XGBoost can handle missing values directly without requiring extensive preprocessing.
Fast and Efficient: It's optimized for speed and memory usage, making it suitable for large datasets.


Challenges it addresses:

Non-linear relationships between features and target variables.

High dimensionality of data (e.g., many features like income, loan amount, credit history, etc.).
Imbalanced datasets.

xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
# Training the model
xgb_model.fit(X_train, y_train)

# Making predictions
y_pred = xgb_model.predict(X_test)


#Evaluation

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

