import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Credit Risk System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* MAIN THEME */
    .stApp { background-color: #000000; color: #ffffff; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }

    /* BENTO BOX CARD */
    .bento-box {
        background-color: #111111;
        border: 1px solid #333;
        border-radius: 24px;
        padding: 25px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .bento-box:hover { border-color: #555; }

    /* TYPOGRAPHY */
    h1, h2, h3 { color: #ffffff !important; font-weight: 700; letter-spacing: -0.5px; }
    p, label, li, span, div { color: #a0a0a0; }

    /* METRICS */
    .metric-value { font-size: 3.5rem; font-weight: 800; color: #fff; line-height: 1; margin: 10px 0; }
    .metric-label { color: #666; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }

    /* PILLS */
    .status-pill { display: inline-block; padding: 6px 16px; border-radius: 50px; font-size: 0.85rem; font-weight: 600; color: #fff !important; }
    .pill-green { background-color: rgba(76, 217, 100, 0.2); color: #4cd964 !important; border: 1px solid rgba(76, 217, 100, 0.3); }
    .pill-red { background-color: rgba(255, 69, 58, 0.2); color: #ff453a !important; border: 1px solid rgba(255, 69, 58, 0.3); }

    /* CUSTOM TABLE STYLES */
    .logic-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    .logic-table th { text-align: left; color: #666; font-size: 0.8rem; text-transform: uppercase; padding-bottom: 10px; border-bottom: 1px solid #333; }
    .logic-table td { padding: 12px 0; border-bottom: 1px solid #222; font-size: 0.95rem; color: #ddd; }
    .logic-table tr:last-child td { border-bottom: none; }

    .impact-bar-bg { width: 100px; height: 8px; background-color: #222; border-radius: 4px; display: inline-block; vertical-align: middle; margin-right: 10px; }
    .impact-bar-fill { height: 100%; border-radius: 4px; }

    .text-green { color: #4cd964; font-weight: 600; }
    .text-red { color: #ff453a; font-weight: 600; }

</style>
""", unsafe_allow_html=True)


# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        pipeline = joblib.load('models/loan_pipeline.pkl')
        with open('models/threshold_config.json', 'r') as f:
            config = json.load(f)
        return pipeline, config['optimal_threshold']
    except:
        return None, None


pipeline, OPTIMAL_THRESHOLD = load_assets()
if pipeline is None:
    st.error("🚨 Models not found. Please run train.py first.")
    st.stop()

# --- HEADER ---
st.markdown("<h1 style='text-align: center; margin-bottom: 40px; font-size: 3rem;'>CREDIT RISK ASSESSMENT SYSTEM</h1>",
            unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("Applicant Profile")
st.sidebar.markdown("---")

gender = "Male"
married = "Yes"

dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
applicant_income = st.sidebar.number_input("Applicant Income (₹ Monthly)", 0, 100000, 5000, step=500)
coapplicant_income = st.sidebar.number_input("Co-Applicant Income (₹ Monthly)", 0, 50000, 0, step=500)
loan_amount = st.sidebar.number_input("Loan Amount Request (₹ Lakhs)", 10, 1000, 120, step=10)
loan_term = st.sidebar.selectbox("Loan Term (Months)", [360.0, 180.0, 480.0, 300.0, 84.0])
credit_history = st.sidebar.radio("Credit Bureau History", [1.0, 0.0],
                                  format_func=lambda x: "Good (No Default)" if x == 1.0 else "Bad (Prior Default)")
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# --- LOGIC ---
input_data = pd.DataFrame({
    'Gender': [gender], 'Married': [married], 'Dependents': [dependents],
    'Education': [education], 'Self_Employed': [self_employed],
    'ApplicantIncome': [applicant_income], 'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount], 'Loan_Amount_Term': [loan_term],
    'Credit_History': [credit_history], 'Property_Area': [property_area],
    'Loan_ID': ['Test'], 'Loan_Status': ['N']
})
input_data['TotalIncome'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
model_input = input_data[['TotalIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
                          'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']]

# Prediction
prob = pipeline.predict_proba(model_input)[0, 1]
decision = "APPROVED" if prob >= OPTIMAL_THRESHOLD else "REJECTED"
color_class = "pill-green" if decision == "APPROVED" else "pill-red"

# --- EXPLAINABILITY ---
preprocessor = pipeline.named_steps['preprocessor']
classifier = pipeline.named_steps['classifier']

X_trans = preprocessor.transform(model_input)
ohe = preprocessor.named_transformers_['cat']
cat_cols = ohe.get_feature_names_out()
clean_cat_cols = [c.replace('x0_', '').replace('_', ' ') for c in cat_cols]
feat_names = ['Total Income', 'Loan Amount', 'Loan Term', 'Credit History'] + clean_cat_cols

coeffs = classifier.coef_[0]
input_row = X_trans[0]
if hasattr(input_row, "toarray"):
    input_row = input_row.toarray()[0]

# Fix dimensions (ravel) to ensure 1D array
impacts = (coeffs * input_row).ravel()

# --- LAYOUT ---
col1, col2 = st.columns([1.3, 2])

with col1:
    st.markdown(
        f"""<div class="bento-box"><div class="metric-label">Risk Decision</div><div><span class="status-pill {color_class}">{decision}</span></div><div class="metric-value">{prob:.1%}</div><div style="color: #666; font-size: 0.8rem;">Approval Probability (Cutoff: {OPTIMAL_THRESHOLD:.1%})</div></div>""",
        unsafe_allow_html=True)

with col2:
    impact_tuples = list(zip(feat_names, impacts))
    filtered_tuples = [x for x in impact_tuples if "Gender" not in x[0] and "Married" not in x[0]]
    sorted_impacts = sorted(filtered_tuples, key=lambda x: abs(x[1]), reverse=True)

    pos_factors = [x for x in sorted_impacts if x[1] > 0]
    neg_factors = [x for x in sorted_impacts if x[1] < 0]

    if decision == "APPROVED":
        primary_factors = pos_factors[:3] if pos_factors else sorted_impacts[:3]
    else:
        primary_factors = neg_factors[:3] if neg_factors else sorted_impacts[:3]

    reason_items = ""
    for name, score in primary_factors:
        if score > 0:
            reason_items += f"<li>✅ <b>{name}</b> strengthens the application.</li>"
        else:
            reason_items += f"<li>⚠️ <b>{name}</b> increases risk.</li>"

    if not reason_items: reason_items = "<li>No dominant risk factors detected.</li>"

    st.markdown(
        f"""<div class="bento-box"><div class="metric-label">Appraisal Summary</div><h3 style="margin-top: 5px;">Key Drivers</h3><ul style='color: #aaa; padding-left: 20px; line-height: 1.6;'>{reason_items}</ul></div>""",
        unsafe_allow_html=True)

# --- DECISION LOGIC TABLE ---
st.markdown("<br>", unsafe_allow_html=True)

# We use a container instead of a column to ensure it renders fully width
with st.container():
    df_viz = pd.DataFrame({'Feature': feat_names, 'Impact': impacts})
    df_viz['AbsImpact'] = df_viz['Impact'].abs()

    # Filtering
    df_viz = df_viz[~df_viz['Feature'].str.contains("Gender")]
    df_viz = df_viz[~df_viz['Feature'].str.contains("Married")]

    max_val = df_viz['AbsImpact'].max()
    if max_val == 0: max_val = 1

    df_viz['Points'] = ((df_viz['Impact'] / max_val) * 100).astype(int)
    df_viz['AbsPoints'] = df_viz['Points'].abs()
    df_viz = df_viz.sort_values('AbsPoints', ascending=False).head(7)
    df_viz['BarWidth'] = df_viz['AbsPoints']

    rows_html = ""
    for index, row in df_viz.iterrows():
        feature = row['Feature']
        points = row['Points']
        bar_width = row['BarWidth']

        if points > 0:
            role = "<span class='text-green'>⬆ Lowers Risk</span>"
            bar_color = "#4cd964"
            points_display = f"+{points}"
        elif points < 0:
            role = "<span class='text-red'>⬇ Increases Risk</span>"
            bar_color = "#ff453a"
            points_display = f"{points}"
        else:
            role = "<span style='color:#666'>- Neutral</span>"
            bar_color = "#333"
            points_display = "0"

        display_width = max(bar_width, 2)

        rows_html += f"<tr><td>{feature}</td><td>{role}</td><td><div class='impact-bar-bg'><div class='impact-bar-fill' style='width: {display_width}%; background-color: {bar_color};'></div></div><span style='font-size: 0.9rem; font-family: monospace; color: #fff;'>{points_display} pts</span></td></tr>"

    table_html = f"""<div class="bento-box" style="height: auto;"><div class="metric-label">Deep Dive</div><h3 style="margin-bottom: 15px;">Risk Factor Scorecard</h3><p style="font-size: 0.9rem; margin-bottom: 20px;">Normalized impact of financial variables on the final credit decision (Max 100 points).</p><table class="logic-table"><thead><tr><th width="35%">Factor</th><th width="20%">Effect</th><th width="45%">Relative Impact</th></tr></thead><tbody>{rows_html}</tbody></table></div>"""

    st.markdown(table_html, unsafe_allow_html=True)