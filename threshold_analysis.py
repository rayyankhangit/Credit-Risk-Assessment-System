import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import confusion_matrix


def analyze_thresholds():
    # Load the trained model and test data
    try:
        model = joblib.load('models/loan_pipeline.pkl')
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv').values.ravel()
    except FileNotFoundError:
        print("ERROR: Could not find model or data. Did you run train.py?")
        return

    # Get the probability scores (0 to 1)
    y_probs = model.predict_proba(X_test)[:, 1]

    # --- BUSINESS RULES ---
    # Cost of a False Positive (Bad Loan) is HIGH (loss of money)
    # Cost of a False Negative (Missed Customer) is LOW (loss of interest)
    COST_FP = 5
    COST_FN = 1

    thresholds = np.arange(0.0, 1.0, 0.01)
    costs = []

    for thresh in thresholds:
        preds = (y_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        total_cost = (fp * COST_FP) + (fn * COST_FN)
        costs.append(total_cost)

    # Find the cheapest threshold
    best_index = np.argmin(costs)
    optimal_threshold = thresholds[best_index]

    print(f">> Optimal Threshold found: {optimal_threshold:.2f}")

    # Save this number for the App to use later
    config = {'optimal_threshold': float(optimal_threshold)}
    with open('models/threshold_config.json', 'w') as f:
        json.dump(config, f)
    print(">> Threshold saved to models/threshold_config.json")


if __name__ == "__main__":
    analyze_thresholds()