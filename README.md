

# Loan Default Prediction App

## Overview

This web app predicts the likelihood of a loan default based on user inputs like Age, Income, Loan Amount, and other relevant features. It leverages three powerful machine learning models: XGBoost, Random Forest, and Deep Learning to provide predictions.

## Features

- **User-friendly Interface**: Enter key details (age, income, loan amount, etc.) via a simple form.
- **Multiple Model Predictions**: Get predictions from three distinct models for robust results:
  - XGBoost
  - Random Forest
  - Deep Learning
- **Real-time Predictions**: Receive immediate loan default predictions based on your input.

## Installation

1. Clone this repository:
   ```bash
   git clone <https://github.com/sharmaraghav644/PrivateRepo>
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## How to Use

1. Enter details about the loan and borrower (e.g., Age, Income, Loan Amount).
2. Select categorical options (e.g., Education, Employment Type).
3. Click Predict to view the loan default likelihood for each model.

## Models Used

- **XGBoost**: A powerful gradient boosting model.
- **Random Forest**: A robust ensemble learning method.
- **Deep Learning**: A neural network-based model with threshold classification.

## Dataset

The model is trained on the Loan Default Prediction Dataset. You can access it here - https://www.kaggle.com/datasets/nikhil1e9/loan-default/data

## Note

The model inputs should closely match the dataset's original features for optimal performance. Default values are used for missing features.
You can also check the requirements.txt file in the repository which includes all the dependencies needed for this model. The list excludes the dependencies which are pre-installed in Streamlit.

## License

MIT License