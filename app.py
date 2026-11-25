import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Main container spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2563eb 100%);
        color: #ffffff;
        padding: 1.75rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.14);
    }
    .hero-banner h1 {
        margin: 0 0 0.4rem 0;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .hero-banner p {
        margin: 0;
        font-size: 1rem;
        color: #dbeafe;
    }

    /* Card containers */
    .section-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
    }

    /* Prediction result boxes */
    .result-churn {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 6px solid #dc2626;
        color: #7f1d1d;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        margin-top: 0.75rem;
    }
    .result-stay {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 6px solid #16a34a;
        color: #14532d;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        margin-top: 0.75rem;
    }
    .result-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .result-sub {
        font-size: 0.95rem;
        margin: 0;
    }

    /* Pill badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background-color: rgba(37, 99, 235, 0.14);
        color: #2563eb;
        margin-bottom: 0.5rem;
    }

    /* Primary button styling */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.28);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Constants Matching Your Notebook & Dataset
# -----------------------------------------------------------------------------
DEFAULT_NUM_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
DEFAULT_CAT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

# -----------------------------------------------------------------------------
# Loaders (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts(pkl_path: str = "customer_churn_artifacts.pkl"):
    if not os.path.exists(pkl_path):
        return None
    return joblib.load(pkl_path)


@st.cache_data
def load_dataset(csv_path: str = "Customer_Churn_Cleaned.csv"):
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def preprocess_features(input_df: pd.DataFrame, scaler, encoder, num_cols: list, cat_cols: list) -> pd.DataFrame:
    """Applies the exact MinMaxScaler + OneHotEncoder concatenation used during training."""
    num_part = input_df[num_cols].copy()
    cat_part = input_df[cat_cols].copy()

    num_scaled = scaler.transform(num_part)
    cat_encoded = encoder.transform(cat_part)

    # Convert to DataFrames if numpy arrays were returned
    if not isinstance(num_scaled, pd.DataFrame):
        num_scaled = pd.DataFrame(num_scaled, columns=num_cols)
    if not isinstance(cat_encoded, pd.DataFrame):
        cat_cols_out = encoder.get_feature_names_out(cat_cols)
        cat_encoded = pd.DataFrame(cat_encoded, columns=cat_cols_out)

    features_df = pd.concat(
        [num_scaled.reset_index(drop=True), cat_encoded.reset_index(drop=True)],
        axis=1,
    )
    return features_df


# -----------------------------------------------------------------------------
# Header & Sidebar
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>📉 Telecom Customer Churn Prediction</h1>
        <p>Interactive prediction dashboard powered by your trained KNN model, MinMaxScaler, and OneHotEncoder artifacts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

artifacts = load_artifacts()
df_ref = load_dataset()

with st.sidebar:
    st.markdown("### ⚙️ Model & Data Setup")
    if artifacts is not None:
        st.success("Loaded `customer_churn_artifacts.pkl`")
    else:
        st.warning("`customer_churn_artifacts.pkl` not found in current folder.")
        uploaded_pkl = st.file_uploader("Upload `.pkl` artifacts", type=["pkl"])
        if uploaded_pkl is not None:
            artifacts = joblib.load(uploaded_pkl)
            st.success("Uploaded `.pkl` loaded!")

    if df_ref is None:
        uploaded_csv = st.file_uploader("Upload `Customer_Churn_Cleaned.csv` (optional)", type=["csv"])
        if uploaded_csv is not None:
            df_ref = pd.read_csv(uploaded_csv)

    st.markdown("---")
    st.markdown("### 📌 Model Summary")
    if artifacts is not None:
        model_obj = artifacts["model"]
        st.write(f"**Algorithm:** `{type(model_obj).__name__}`")
        if hasattr(model_obj, "n_neighbors"):
            st.write(f"**Neighbors (`k`):** `{model_obj.n_neighbors}`")
        st.write(f"**Input Features:** `{len(DEFAULT_NUM_COLS) + len(DEFAULT_CAT_COLS)}` raw → `{getattr(model_obj, 'n_features_in_', 'N/A')}` encoded")
    st.caption("Label mapping: `0` = No (Stay), `1` = Yes (Churn)")

if artifacts is None:
    st.error("Please place `customer_churn_artifacts.pkl` in the same folder as `app.py` or upload it in the sidebar to run predictions.")
    st.stop()

model = artifacts["model"]
scaler = artifacts["scaler"]
encoder = artifacts["encoder"]
num_cols = artifacts.get("num_cols", DEFAULT_NUM_COLS)
cat_cols = artifacts.get("cat_cols", DEFAULT_CAT_COLS)

# -----------------------------------------------------------------------------
# Main Tabs
# -----------------------------------------------------------------------------
tab_single, tab_batch, tab_data = st.tabs(
    ["🎯 Single Customer Prediction", "📂 Batch CSV Prediction", "📊 Dataset Explorer"]
)

# =============================================================================
# TAB 1: Single Customer Prediction
# =============================================================================
with tab_single:
    with st.form("churn_prediction_form"):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<span class="badge">ACCOUNT & BILLING</span>', unsafe_allow_html=True)
        st.subheader("Customer Profile & Charges")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            customer_id = st.text_input("Customer ID (optional)", value="7590-VHVEG")
            senior_label = st.selectbox("Senior Citizen", options=["No (0)", "Yes (1)"], index=0)
            SeniorCitizen = 1 if senior_label.startswith("Yes") else 0
        with c2:
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12, step=1)
            Contract = st.selectbox("Contract Type", options=CATEGORY_OPTIONS["Contract"], index=0)
        with c3:
            MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.35, step=0.5)
            PaperlessBilling = st.selectbox("Paperless Billing", options=CATEGORY_OPTIONS["PaperlessBilling"], index=1)
        with c4:
            TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, value=1394.55, step=5.0)
            PaymentMethod = st.selectbox("Payment Method", options=CATEGORY_OPTIONS["PaymentMethod"], index=2)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<span class="badge">DEMOGRAPHICS & SERVICES</span>', unsafe_allow_html=True)
        st.subheader("Demographics & Subscribed Services")

        s1, s2, s3 = st.columns(3)
        with s1:
            gender = st.selectbox("Gender", options=CATEGORY_OPTIONS["gender"], index=0)
            Partner = st.selectbox("Partner", options=CATEGORY_OPTIONS["Partner"], index=0)
            Dependents = st.selectbox("Dependents", options=CATEGORY_OPTIONS["Dependents"], index=0)
            PhoneService = st.selectbox("Phone Service", options=CATEGORY_OPTIONS["PhoneService"], index=1)
            MultipleLines = st.selectbox("Multiple Lines", options=CATEGORY_OPTIONS["MultipleLines"], index=0)

        with s2:
            InternetService = st.selectbox("Internet Service", options=CATEGORY_OPTIONS["InternetService"], index=1)
            OnlineSecurity = st.selectbox("Online Security", options=CATEGORY_OPTIONS["OnlineSecurity"], index=0)
            OnlineBackup = st.selectbox("Online Backup", options=CATEGORY_OPTIONS["OnlineBackup"], index=0)
            DeviceProtection = st.selectbox("Device Protection", options=CATEGORY_OPTIONS["DeviceProtection"], index=0)
            TechSupport = st.selectbox("Tech Support", options=CATEGORY_OPTIONS["TechSupport"], index=0)

        with s3:
            StreamingTV = st.selectbox("Streaming TV", options=CATEGORY_OPTIONS["StreamingTV"], index=0)
            StreamingMovies = st.selectbox("Streaming Movies", options=CATEGORY_OPTIONS["StreamingMovies"], index=0)

        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Churn Risk", type="primary")

    if submitted:
        row_dict = {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        }
        single_df = pd.DataFrame([row_dict])
        X_ready = preprocess_features(single_df, scaler, encoder, num_cols, cat_cols)

        pred_class = int(model.predict(X_ready)[0])
        churn_prob = float(model.predict_proba(X_ready)[0][1]) if hasattr(model, "predict_proba") else float(pred_class)
        stay_prob = 1.0 - churn_prob

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Probability", f"{churn_prob * 100:.1f}%")
        m2.metric("Retention Probability", f"{stay_prob * 100:.1f}%")
        m3.metric("Predicted Class", "Yes (Churn)" if pred_class == 1 else "No (Stay)")

        if pred_class == 1:
            st.markdown(
                f"""
                <div class="result-churn">
                    <div class="result-title">⚠️ High Churn Risk Detected ({customer_id})</div>
                    <p class="result-sub">The model predicts this customer is likely to <strong>churn</strong> (estimated probability: <strong>{churn_prob * 100:.1f}%</strong>). Consider targeted retention offers or contract incentives.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-stay">
                    <div class="result-title">✅ Low Churn Risk — Likely to Stay ({customer_id})</div>
                    <p class="result-sub">The model predicts this customer is likely to <strong>remain active</strong> (estimated churn probability: <strong>{churn_prob * 100:.1f}%</strong>).</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =============================================================================
# TAB 2: Batch CSV Prediction
# =============================================================================
with tab_batch:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Upload Customer CSV for Bulk Scoring")
    st.write("Upload a CSV containing the feature columns (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, and the 15 categorical columns). `Churn` and `customerID` columns are optional.")

    batch_file = st.file_uploader("Choose a CSV file", type=["csv"], key="batch_uploader")

    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        required_cols = list(dict.fromkeys(num_cols + cat_cols))
        missing_cols = [c for c in required_cols if c not in batch_df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            X_batch = preprocess_features(batch_df, scaler, encoder, num_cols, cat_cols)
            preds = model.predict(X_batch)
            probs = model.predict_proba(X_batch)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

            out_df = batch_df.copy()
            out_df["Churn_Prediction"] = np.where(preds == 1, "Yes", "No")
            out_df["Churn_Probability"] = np.round(probs, 4)

            c_b1, c_b2, c_b3 = st.columns(3)
            c_b1.metric("Total Customers Scored", f"{len(out_df):,}")
            c_b2.metric("Predicted Churners ('Yes')", f"{int((preds == 1).sum()):,}")
            c_b3.metric("Avg. Churn Probability", f"{float(np.mean(probs)) * 100:.1f}%")

            st.dataframe(out_df.head(50), use_container_width=True)

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_bytes,
                file_name="churn_predictions.csv",
                mime="text/csv",
                type="primary",
            )
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 3: Dataset Explorer
# =============================================================================
with tab_data:
    if df_ref is not None:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Cleaned Dataset Overview (`Customer_Churn_Cleaned.csv`)")
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Rows", f"{len(df_ref):,}")
        d2.metric("Total Columns", f"{df_ref.shape[1]}")
        if "Churn" in df_ref.columns:
            churn_rate = (df_ref["Churn"] == "Yes").mean() * 100
            d3.metric("Historical Churn Rate", f"{churn_rate:.1f}%")

        st.dataframe(df_ref.head(25), use_container_width=True)

        if "Contract" in df_ref.columns and "Churn" in df_ref.columns:
            st.markdown("#### Churn Rate by Contract Type")
            contract_summary = (
                df_ref.groupby("Contract")["Churn"]
                .apply(lambda s: round((s == "Yes").mean() * 100, 1))
                .reset_index(name="Churn Rate (%)")
            )
            st.bar_chart(contract_summary.set_index("Contract"))
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Place `Customer_Churn_Cleaned.csv` in the app directory or upload it in the sidebar to explore the dataset.")