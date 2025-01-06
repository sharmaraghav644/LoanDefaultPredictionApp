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

# Load dataset
file_path = "/Users/raghavsharma/desktop/loan_default_predication_kaggle.csv"
df = pd.read_csv(file_path)

# Print the column names
print(df.columns)

# Print the shape of the dataset
print(f"Shape of the dataset: {df.shape}")

# Display first 5 rows
print(df.head(5))

# Data types and missing values
print(df.info())

# Summary statistics
print(df.describe())
print(df.isnull().sum())

# Calculate missing value percentage
missing_percent = df.isnull().sum() / len(df) * 100
print("Missing value percentage:\n", missing_percent)

# Distribution of the Target Variable (Default)
sns.countplot(x='Default', data=df)
plt.title('Distribution of Loan Default Status')
plt.show()

# Group by 'Default' to see the counts of each
default_grouped = df.groupby('Default').size()
print(default_grouped)

# Univariate Analysis of Continuous Features
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for Income Distribution
sns.histplot(df['Income'], kde=True, ax=axes[0], color='blue')
axes[0].set_title('Income Distribution')
axes[0].set_xlabel('Income')
axes[0].set_ylabel('Frequency')

# Plot for Loan Amount Distribution
sns.histplot(df['LoanAmount'], kde=True, ax=axes[1], color='green')
axes[1].set_title('Loan Amount Distribution')
axes[1].set_xlabel('Loan Amount')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Boxplot of Income by Loan Default Status
sns.boxplot(x='Default', y='Income', data=df)
plt.title('Applicant Income by Loan Default Status')
plt.show()

# Analysis of Categorical Features
sns.countplot(x='MaritalStatus', data=df)
plt.title('Marital Status Distribution')
plt.show()

sns.countplot(x='Education', data=df)
plt.title('Education Distribution')
plt.show()

# Correlation heatmap
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Pairwise relationships plot
sns.pairplot(df[['Income', 'LoanAmount', 'InterestRate', 'Default']], hue='Default')
plt.title('Pairwise Relationships')
plt.show()

# Drop LoanID as it's not useful for the prediction
df.drop(columns=['LoanID'], inplace=True)

# Encoding categorical columns using LabelEncoder
label_encoder = LabelEncoder()

# List of categorical columns
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']

# Encode each categorical column and drop the original columns
for col in categorical_cols:
    df[col + '_encoded'] = label_encoder.fit_transform(df[col])

df.drop(columns=categorical_cols, inplace=True)

# Verify the data types again to ensure they are numeric
print(df.dtypes)

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
y_pred = xgb_model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)

accuracy = accuracy_score(y_test, y_pred1)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred1))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred1))