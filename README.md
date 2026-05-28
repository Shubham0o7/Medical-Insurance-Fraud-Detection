# 🏥 Intelligent Healthcare Fraud Detection Platform

An end-to-end Deep Learning Analytics framework designed to identify anomalous and potentially fraudulent billing patterns across healthcare provider networks. This repository features a high-performance Artificial Neural Network (ANN) trained on statistically optimized provider aggregates and deployed via a real-time Streamlit executive dashboard.

---

## 🚀 Key Features

- **Algorithmic Feature Optimization**: Engineered a clean, human-in-the-loop interface by shifting away from high-dimensional uninterpretable matrices. Implemented a Pearson Correlation analysis to filter 131 sparse dataset dimensions down to the Top 10 Most Statistically Correlated Risk Predictors, heavily maximizing model focus on provider-cluster anomalies.
- **Optimized Deep Learning Core**: Formulated and trained a Multi-Layer Perceptron (MLP) Sequential Neural Network natively using TensorFlow and Keras. By eliminating PCA-bottleneck layers and structural autoencoders, the model processes pure, unwarped feature distributions to output real-time binary fraud probabilities.
- **Production-Grade Data Serialization**: Automated full runtime data pipeline stability by utilizing `scikit-learn` Standard Scalers. Pre-fitted training metrics are serialized via `joblib`, enabling live, interactive input parameters to be instantly normalized without feature name or dimensional tracking errors.
- **High-Fidelity Executive Dashboard**: Deployed using a modern Streamlit frontend structured logically across operational paradigms (*Transaction & Resource Volume*, *Financial & Care Metrics*, and *Diagnostic Profiling*). The interface includes embedded tooltips, contextual fallback loops, and dynamic visual risk alerts to simulate an enterprise-level risk auditing platform.

---

## 🛠️ Tech Stack & Ecosystem

- **Frontend / Dashboard**: Streamlit
- **Machine Learning Core**: TensorFlow (v2.x), Keras
- **Data Engineering & Analysis**: Python, Pandas, NumPy, Scipy
- **Pipeline Serialization**: Joblib, Scikit-Learn

---

## 📊 Feature Selection Strategy (Top 10 Metrics)
To build a highly actionable real-world tool, the network discards raw patient metrics and targets **Provider-Cluster Aggregates** that exhibit the strongest statistical correlation with verified fraud distributions:

1. **Total Insurance Claims Filed** (`ClmCount_Provider` - *Corr: 0.5155*)
2. **Primary Physician Caseload Count** (`ClmCount_Provider_AttendingPhysician` - *Corr: 0.3005*)
3. **Average Inpatient Length of Stay (Days)** (`PerProviderAvg_AdmitForDays` - *Corr: 0.2925*)
4. **Secondary/Consultant Physician Caseload** (`ClmCount_Provider_OtherPhysician` - *Corr: 0.1932*)
5. **Average Claim Reimbursement ($)** (`PerProviderAvg_InscClaimAmtReimbursed` - *Corr: 0.1909*)
6. **Secondary Diagnosis Core Frequency Flag** (`ClmCount_Provider_ClmDiagnosisCode_2` - *Corr: 0.1905*)
7. **Average Patient Deductible Paid ($)** (`PerProviderAvg_DeductibleAmtPaid` - *Corr: 0.1844*)
8. **Tertiary Diagnosis Complex Frequency Flag** (`ClmCount_Provider_ClmDiagnosisCode_3` - *Corr: 0.1587*)
9. **Average Annual Inpatient Reimbursement ($)** (`PerProviderAvg_IPAnnualReimbursementAmt` - *Corr: 0.1564*)
10. **Unique Patient Beneficiary Base** (`ClmCount_Provider_BeneID` - *Corr: 0.1557*)

---

## 📥 Model & Scaler Downloads
To adhere to software engineering production standards and prevent repository bloat, large binary artifacts are kept out of the Git tracking history. Download the serialized pipeline files from the link below and place them directly into your project's root directory:

- [🔗 Download Serialized Pipeline Components (Google Drive)](https://drive.google.com/drive/folders/1LR_JPgoE9nfJukxHnGDwlp2MzAWatJeF?usp=drive_link)

---

## 📦 Project Architecture & Directory Layout

```text
├── Medical_Insurance_Fraud_Detection.ipynb   # Google Colab Training Notebook
├── app.py                                   # Streamlit Production Application
├── medical_fraud_model.h5                   # Trained Keras Neural Network Weights
├── sc.pkl                                   # Serialized 10-Feature Standard Scaler
├── .gitignore                               # Specifies intentionally untracked files
└── README.md                                # Platform Documentation

```

---

## 💻 Installation & Local Deployment

Follow these quick steps to cleanly launch the ecosystem locally on your system:

### 1. Clone the Repository
```bash
git clone https://github.com/shubham0o7/medical-insurance-fraud-detection.git
cd medical-insurance-fraud-detection

```

### 2. Install Required Dependencies

```bash
pip install -r requirements.txt

```

### 3. Add Binary Pipeline Components

Download your newly generated medical_fraud_model.h5 and sc.pkl components from your secure cloud links and drop them straight into the root project directory alongside your app.py script.

### 4. Run the Production Server
```bash
streamlit run app.py

```

A local browser window will instantly deploy at http://localhost:8501.

---

## 🔍 Analytical Inference Pipeline Logic
When a user executes a check inside the user interface, the application processes inputs using the following sequential workflow:
* **📥 Matrix Capture:** Collects the 10 domain metrics via the structured web dashboard fields.
* **📐 Vector Alignment:** Converts raw parameters into a 2D NumPy array, guaranteeing they precisely match the correlation-ranked index sequence.
* **⚖️ Statistical Scaling:** Passes the array through sc.pkl to normalize scale variances seamlessly based on training data attributes.
* **🧠 Deep Network Evaluation:** Feeds the normalized vectors directly into medical_fraud_model.h5 without dimensional tracking errors.
* **🎯 Verdict Generation:** Passes the output probability scalar through a $0.5$ decision boundary threshold to trigger immediate COMPLIANT BASELINE or CRITICAL: AUDIT REQUIRED visual alerts.
