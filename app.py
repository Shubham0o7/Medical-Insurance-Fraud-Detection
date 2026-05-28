import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Healthcare Fraud Analytics Dashboard",
    page_icon="🏥",
    layout="centered"
)

# 2. Cached Pipeline Components Loader
@st.cache_resource
def load_fraud_pipeline():
    model = tf.keras.models.load_model("medical_fraud_model.h5")
    sc = joblib.load("sc.pkl")
    return model, sc

try:
    model, sc = load_fraud_pipeline()
    pipeline_loaded = True
except Exception as config_err:
    st.error(f"🚨 Configuration Loading Error: {config_err}")
    pipeline_loaded = False

# 3. Form Header & UI Typography
st.title("🏥 Intelligent Healthcare Fraud Detection Platform")
st.markdown("Advanced risk-scoring framework tracking anomalous behavior distributions across provider clusters.")
st.write("---")

st.subheader("📋 Operational Risk Metrics Engine")
st.info("Adjust the provider metrics below to evaluate behavioral correlation scores in real-time.")

# 4. Interactive User Input Form with Clean Professional Labels
with st.form(key="optimized_fraud_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📊 Transaction & Resource Volume")
        f1 = st.number_input("Total Insurance Claims Filed", min_value=0, value=150, step=10, help="Total historical claims associated with this specific provider.")
        f2 = st.number_input("Primary Physician Caseload Count", min_value=0, value=40, step=5, help="Number of distinct claims involving the provider and their main attending physicians.")
        f4 = st.number_input("Consultant Physician Caseload Count", min_value=0, value=15, step=5, help="Number of distinct claims involving other associated medical staff.")
        f10 = st.number_input("Unique Patient Beneficiary Base", min_value=0, value=110, step=10, help="Total unique patients treated by this provider entity.")
        
    with col2:
        st.markdown("##### 💰 Financial & Hospitality Metrics")
        f5 = st.number_input("Average Claim Reimbursement ($)", min_value=0.0, value=3500.0, step=100.0, help="The mean dollar amount requested back from insurance per claim.")
        f7 = st.number_input("Average Patient Deductible Paid ($)", min_value=0.0, value=1050.0, step=50.0, help="The mean out-of-pocket patient copay or deductible threshold recorded.")
        f9 = st.number_input("Average Annual Inpatient Reimbursement ($)", min_value=0.0, value=15000.0, step=500.0, help="Aggregated annual institutional inpatient tracking average.")
        f3 = st.number_input("Average Duration of Stay (Days)", min_value=0.0, value=4.5, step=0.5, help="The calculated mean duration of stay recorded across patient clusters.")

    st.write("---")
    st.markdown("##### 🧬 Diagnostic Profiling")
    cc_col1, cc_col2 = st.columns(2)
    with cc_col1:
        f6 = st.number_input("Secondary Diagnosis Core Frequency Flag", min_value=0, value=25, step=5, help="Provider grouping count linked directly to chronic tracking codes.")
    with cc_col2:
        f8 = st.number_input("Tertiary Diagnosis Complex Frequency Flag", min_value=0, value=10, step=2, help="Provider grouping count linked to high-level complex medical codes.")

    st.write("")
    submit_button = st.form_submit_button(label="🚀 Run Risk Diagnostic")

# 5. Live Inference Stream
if submit_button and pipeline_loaded:
    with st.spinner("Analyzing operational matrix vectors..."):
        try:
            # The background array maintains the exact 1-10 statistical ranking your model needs
            raw_features = np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]], dtype=np.float32)
            
            # Step 1: Scale features using your updated 10-column StandardScaler
            scaled_features = sc.transform(raw_features)
            
            # Step 2: Run inference on your newly trained ANN model
            fraud_probability = model.predict(scaled_features)[0][0]
            
            # 6. Results Interface Rendering
            st.write("---")
            st.subheader("🔍 Automated Forensic Output")
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric(label="Risk Correlation Index", value=f"{fraud_probability * 100:.2f}%")
            with m2:
                status_label = "CRITICAL: AUDIT REQUIRED" if fraud_probability >= 0.5 else "COMPLIANT BASELINE"
                st.metric(label="System Assessment Verdict", value=status_label)

            if fraud_probability >= 0.5:
                st.error("⚠️ **High-Risk Behavioral Signature Detected:** This provider network shows strong statistical correlations with historical billing irregularities and overutilization patterns.")
            else:
                st.success("✅ **Standard Operational Baseline:** Provider behavior falls completely within typical compliant healthcare delivery metrics.")
                
        except Exception as pred_error:
            st.error(f"Inference Failure: {pred_error}")