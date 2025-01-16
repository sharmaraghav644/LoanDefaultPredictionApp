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

missing_percent = df.isnull().sum() / len(df) * 100
print(missing_percent)


sns.countplot(x='Default', data=df)
plt.title('Distribution of Loan Default Status')
plt.show()

default_grouped = df.groupby('Default').size()

#0 indicates not default (good)
#1 indicates default (bad)

# Print the result
print(default_grouped)


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


sns.countplot(x='MaritalStatus', data=df)
plt.title('Marital Status Distribution')
plt.show()

sns.countplot(x='Education', data=df)
plt.title('Education Distribution')
plt.show()


corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


sns.pairplot(df[['Income', 'LoanAmount', 'InterestRate', 'Default']], hue='Default')
plt.title('Pairwise Relationships')
plt.show()


df.drop(columns=['LoanID'], inplace=True)

# Group by education and calculate default rate
education_default_rate = df.groupby('Education')['Default'].mean().sort_values(ascending=False)
print("Default Rate by Education Level:\n", education_default_rate)

# Visualization
education_default_rate.plot(kind='bar', title='Default Rate by Education Level', color='skyblue')
plt.xlabel('Education Level')
plt.ylabel('Default Rate')
plt.show()

# Create income bins
df['IncomeBracket'] = pd.cut(df['Income'], bins=[0, 20000, 40000, 60000, 80000, 100000, np.inf], 
                             labels=['0-20k', '20-40k', '40-60k', '60-80k', '80-100k', '100k+'])

# Group by income bracket
income_default_rate = df.groupby('IncomeBracket')['Default'].mean()
print("Default Rate by Income Bracket:\n", income_default_rate)

# Visualization
income_default_rate.plot(kind='bar', title='Default Rate by Income Bracket', color='orange')
plt.xlabel('Income Bracket')
plt.ylabel('Default Rate')
plt.show()

# Create loan amount bins
df['LoanAmountBracket'] = pd.cut(df['LoanAmount'], bins=[0, 5000, 10000, 20000, 30000, 40000, np.inf],
                                 labels=['0-5k', '5-10k', '10-20k', '20-30k', '30-40k', '40k+'])

# Group by loan amount bracket
loan_default_rate = df.groupby('LoanAmountBracket')['Default'].mean()
print("Default Rate by Loan Amount Bracket:\n", loan_default_rate)

# Visualization
loan_default_rate.plot(kind='bar', title='Default Rate by Loan Amount', color='green')
plt.xlabel('Loan Amount Bracket')
plt.ylabel('Default Rate')
plt.show()

# Group by employment type
employment_default_rate = df.groupby('EmploymentType')['Default'].mean().sort_values(ascending=False)
print("Default Rate by Employment Type:\n", employment_default_rate)

# Visualization
employment_default_rate.plot(kind='bar', title='Default Rate by Employment Type', color='purple')
plt.xlabel('Employment Type')
plt.ylabel('Default Rate')
plt.show()

# Group by co-signer presence
cosigner_default_rate = df.groupby('HasCoSigner')['Default'].mean()
print("Default Rate by Co-Signer Presence:\n", cosigner_default_rate)

# Visualization
cosigner_default_rate.plot(kind='bar', title='Default Rate by Co-Signer Presence', color='red')
plt.xlabel('Has Co-Signer')
plt.ylabel('Default Rate')
plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
plt.show()

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


# In[32]:


df = df.apply(pd.to_numeric, errors='coerce')

# Verify data types again
print(df.dtypes)


# Processing and modelling the data

# In[38]:


X = df.drop('Default', axis=1) #feature variables
y = df['Default'] #Target Variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[94]:


xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)

# Training the model
xgb_model.fit(X_train, y_train)

# Making predictions
y_pred_xgb = xgb_model.predict(X_test)


# Evaluating the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Accuracy: {accuracy_xgb * 100:.2f}%\n")

classification_report_xgb = classification_report(y_test, y_pred_xgb)
print("Classification Report:")
print(classification_report_xgb)

confusion_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix_xgb)


smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

xgb_model_smote = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)


# Training the model
xgb_model_smote.fit(X_train_smote, y_train_smote)

# Making predictions
y_pred_xgb_smote = xgb_model_smote.predict(X_test)


# Evaluating the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb_smote)
print(f"Accuracy: {accuracy_xgb * 100:.2f}%\n")

classification_report_xgb_smote = classification_report(y_test, y_pred_xgb_smote)
print("Classification Report:")
print(classification_report_xgb_smote)

confusion_matrix_xgb_smote = confusion_matrix(y_test, y_pred_xgb_smote)
# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix_xgb_smote)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {accuracy_rf * 100:.2f}%\n")

classification_report_rf = classification_report(y_test, y_pred_rf)
print("Classification Report:")
print(classification_report_rf)

confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix_rf)


# Using SMOTE


rf_model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_smote.fit(X_train_smote, y_train_smote)
y_pred_rf_smote = rf_model_smote.predict(X_test)

accuracy_rf_smote = accuracy_score(y_test, y_pred_rf_smote)
print(f"Accuracy: {accuracy_rf_smote * 100:.2f}%\n")

classification_report_rf_smote = classification_report(y_test, y_pred_rf_smote)
print("Classification Report:")
print(classification_report_rf_smote)

confusion_matrix_rf_smote = confusion_matrix(y_test, y_pred_rf_smote)
# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix_rf_smote)



# Build the deep learning model
model_dl = Sequential()
model_dl.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model_dl.add(Dense(32, activation='relu'))
model_dl.add(Dense(1, activation='sigmoid'))

# Compile the model
model_dl.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model_dl.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# In[84]:


# Predict the labels for the test set
y_pred_dl = model_dl.predict(X_test)
y_pred_dl = (y_pred_dl > 0.5).astype(int)  # Convert probabilities to binary labels

# Calculate Accuracy
accuracy_dl = accuracy_score(y_test, y_pred_dl)
print(f"Accuracy: {accuracy_dl * 100:.2f}%\n")

# Classification Report
classification_report_dl = classification_report(y_test, y_pred_dl)
print("Classification Report:")
print(classification_report_dl)

# Confusion Matrix
confusion_matrix_dl = confusion_matrix(y_test, y_pred_dl)
print("Confusion Matrix:")
print(confusion_matrix_dl)


# Using SMOTE

# In[100]:


model_dl_smote = Sequential()
model_dl_smote.add(Dense(64, input_dim=X_train_smote.shape[1], activation='relu'))
model_dl_smote.add(Dense(32, activation='relu'))
model_dl_smote.add(Dense(1, activation='sigmoid'))

# Compile the model
model_dl_smote.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model_dl_smote.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# In[102]:


# Predict the labels for the test set
y_pred_dl_smote = model_dl_smote.predict(X_test)
y_pred_dl_smote = (y_pred_dl_smote > 0.5).astype(int)  # Convert probabilities to binary labels

# Calculate Accuracy
accuracy_dl_smote = accuracy_score(y_test, y_pred_dl_smote)
print(f"Accuracy: {accuracy_dl * 100:.2f}%\n")

# Classification Report
classification_report_dl_smote = classification_report(y_test, y_pred_dl_smote)
print("Classification Report:")
print(classification_report_dl_smote)

# Confusion Matrix
confusion_matrix_dl_smote = confusion_matrix(y_test, y_pred_dl_smote)
print("Confusion Matrix:")
print(confusion_matrix_dl_smote)


# We can now compare the results from all the models - XGB Boost Model, RandomForest Model, Deep Learning Model

# In[86]:


# Create a DataFrame for comparison
comparison_table = pd.DataFrame({
    "Metric": ["Accuracy", "Classification Report", "Confusion Matrix"],
    "XGB Model": [accuracy_xgb, classification_report_xgb, confusion_matrix_xgb],
    "Random Forest Model": [accuracy_rf, classification_report_rf, confusion_matrix_rf],
    "Deep Learning Model": [accuracy_dl, classification_report_dl, confusion_matrix_dl]
})

# Format the table using tabulate
print(tabulate(comparison_table, headers="keys", tablefmt="pretty"))

metrics = {
    "Accuracy": [
        accuracy_xgb,
        accuracy_rf,
        accuracy_dl
    ],
    "Precision": [
        precision_score(y_test, y_pred_xgb, average="binary"),
        precision_score(y_test, y_pred_rf, average="binary"),
        precision_score(y_test, y_pred_dl, average="binary")
    ],
    "Recall": [
        recall_score(y_test, y_pred_xgb, average="binary"),
        recall_score(y_test, y_pred_rf, average="binary"),
        recall_score(y_test, y_pred_dl, average="binary")
    ],
    "F1-Score": [
        f1_score(y_test, y_pred_xgb, average="binary"),
        f1_score(y_test, y_pred_rf, average="binary"),
        f1_score(y_test, y_pred_dl, average="binary")
    ]
}

models = ["XGB Model", "Random Forest Model", "Deep Learning Model"]

# Confusion Matrices
conf_matrices = [confusion_matrix_xgb, confusion_matrix_rf, confusion_matrix_dl]
titles = ["XGB Model", "Random Forest Model", "Deep Learning Model"]

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Model Comparison", fontsize=16)

# Bar chart for metrics
x = np.arange(len(models))  # X-axis positions
width = 0.2  # Width of bars

for i, (metric, values) in enumerate(metrics.items()):
    axes[0, 0].bar(x + i * width, values, width, label=metric)

axes[0, 0].set_title("Performance Metrics")
axes[0, 0].set_xticks(x + width)
axes[0, 0].set_xticklabels(models)
axes[0, 0].set_ylabel("Score")
axes[0, 0].legend()
axes[0, 0].grid(axis="y", linestyle="--", alpha=0.7)

# Heatmaps for confusion matrices
for i, (conf_matrix, title) in enumerate(zip(conf_matrices, titles)):
    row, col = divmod(i + 1, 2)  # Position in subplot grid (1-based index)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[row, col])
    axes[row, col].set_title(f"Confusion Matrix: {title}")
    axes[row, col].set_xlabel("Predicted Label")
    axes[row, col].set_ylabel("True Label")

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Save the trained models as .pkl files
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(dl_model, "dl_model.pkl")


print("Models have been saved successfully!")

