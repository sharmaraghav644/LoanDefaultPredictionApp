import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import bz2

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

## Apply SMOTE to the training data for deep learning model
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Standardize the data
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Build the Deep Learning Model
model_dl = Sequential([
    Input(shape=(X_train.shape[1],)),  # Explicit input layer
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(1, activation='sigmoid')  # Ensure binary classification
])
model_dl.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

# Fit the DL model with SMOTE-resampled data
model_dl.fit(X_train_smote, y_train_smote, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Save the models

model_dl.save("dl_model.keras")  # Save as the recommended .keras format

# Evaluate DL Model
test_loss, test_accuracy = model_dl.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy:.2%}, Test Loss: {test_loss:.4f}")

# Deep Learning Model Predictions
y_pred_dl_prob = model_dl.predict(X_test_scaled)  # Get probabilities
y_pred_dl = (y_pred_dl_prob > 0.5).astype(int)  # Convert probabilities to binary labels

# Calculate Accuracy for DL model
accuracy_dl = accuracy_score(y_test, y_pred_dl)
print(f"Accuracy: {accuracy_dl * 100:.2f}%\n")

# Save models successfully message
print("Deep Learning model has been saved successfully!")

# Confusion Matrix for Deep Learning Model
confusion_matrix_dl = confusion_matrix(y_test, y_pred_dl)
print("Confusion Matrix:")
print(confusion_matrix_dl)

# Manual Prediction Check
for i in range(5):
    print(f"Test {i+1}: {model_dl.predict(X_test_scaled[i].reshape(1, -1))}")
