import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_model():
    # --- 1. Load Data ---
    csv_path = 'data/loan_data_large.csv'
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"ERROR: I cannot find the file at: {csv_path}")
        print("Please make sure 'loan_data.csv' is inside the 'data' folder.")
        return

    df = pd.read_csv(csv_path)
    print(f">> Loaded data: {df.shape}")

    # --- 2. Data Cleaning ---
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    X = df.drop(['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'Loan_Status'], axis=1)
    y = df['Loan_Status']

    # --- 3. Train ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_features = ['TotalIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    print(">> Training model...")
    pipeline.fit(X_train, y_train)

    score = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    print(f">> ROC-AUC Score: {score:.4f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/loan_pipeline.pkl')
    X_test.to_csv('data/X_test.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    print(">> Model saved successfully.")


if __name__ == "__main__":
    train_model()