import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from imblearn.over_sampling import SMOTE
import joblib
import bz2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Function to save compressed models (only for RF and XGBoost)
def save_model_compressed(model, filename):
    with bz2.BZ2File(filename, 'wb') as f:
        joblib.dump(model, f)

# Function to load compressed models (only for RF and XGBoost)
def load_compressed_model(filename):
    with bz2.BZ2File(filename, 'rb') as f:
        return joblib.load(f)

# Load dataset
file_path = "/Users/raghavsharma/desktop/loan_default_predication_kaggle.csv"
df = pd.read_csv(file_path)

# Drop LoanID
df.drop(columns=["LoanID"], inplace=True)

# Encode categorical variables
categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
label_encoder = LabelEncoder()

for col in categorical_cols:
    df[col + '_encoded'] = label_encoder.fit_transform(df[col])

df.drop(columns=categorical_cols, inplace=True)

# Ensure target variable is binary (0 or 1)
df['Default'] = df['Default'].astype(int)

# Split data
X = df.drop('Default', axis=1)
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Calculate class weights for handling imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Train Random Forest with class weights
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost with class weights
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Train Deep Learning Model with class weights
model_dl = Sequential([
    Input(shape=(X_train.shape[1],)),  # Explicit input layer
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Ensure binary classification
])
model_dl.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the DL model with class weights
model_dl.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, class_weight=class_weights_dict)

# Evaluate DL Model
test_loss, test_accuracy = model_dl.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2%}, Test Loss: {test_loss:.4f}")

# Save models correctly
save_model_compressed(xgb_model, "xgb_model_compressed.pkl.bz2")
save_model_compressed(rf_model, "rf_model_compressed.pkl.bz2")
model_dl.save("dl_model.keras")  # Correct way to save deep learning models

print("Models have been saved successfully!")

# Deep Learning Model Predictions
y_pred_dl_prob = model_dl.predict(X_test)  # Get probabilities
y_pred_dl = (y_pred_dl_prob > 0.5).astype(int)  # Convert probabilities to binary labels

# Calculate Accuracy
accuracy_dl = accuracy_score(y_test, y_pred_dl)
print(f"Accuracy: {accuracy_dl * 100:.2f}%\n")

confusion_matrix_dl = confusion_matrix(y_test, y_pred_dl)
print("Confusion Matrix:")
print(confusion_matrix_dl)