# Credit Risk Assessment System 🏦

A machine learning-based decision support system designed to assist banking credit officers in assessing loan applications. This project simulates the credit appraisal process by calculating risk probabilities and providing explainable AI (XAI) insights.

## Key Features
*   **Risk Scoring:** Calculates approval probability based on financial metrics.
*   **Automated Decisioning:** Recommends "Approve" or "Reject" based on optimized thresholds.
*   **Explainable AI:** Uses logical attribution to explain *why* a decision was made (e.g., "High Income lowers risk").
*   **Asset Quality Focus:** Custom threshold tuning to minimize Non-Performing Assets (NPAs).

## Tech Stack
*   **Python**: Core Logic
*   **Scikit-Learn**: Machine Learning (Logistic Regression)
*   **Streamlit**: Interactive Dashboard UI
*   **Pandas/Numpy**: Data Processing

## How to Run Locally
1.  Install dependencies: `pip install -r requirements.txt`
2.  Train the model: `python train.py`
3.  Optimize thresholds: `python threshold_analysis.py`
4.  Launch the dashboard: `streamlit run app.py`
