import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — light mint/teal theme ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --mint-50:  #f0faf8;
    --mint-100: #d6f5ef;
    --mint-200: #aeeade;
    --mint-400: #3dbfa4;
    --mint-500: #23a88e;
    --mint-600: #1a8f78;
    --teal-700: #156659;
    --sky-100:  #e0f4f8;
    --sky-400:  #38b2d4;
    --sky-500:  #1e9dbf;
    --neutral-50:  #f8fafb;
    --neutral-100: #eef1f3;
    --neutral-200: #dde3e7;
    --neutral-500: #6b7c85;
    --neutral-700: #334a55;
    --neutral-900: #0f1f26;
    --danger:   #e5534b;
    --success:  #1a8f78;
    --shadow-sm: 0 1px 3px rgba(21,102,89,.08), 0 1px 2px rgba(21,102,89,.06);
    --shadow-md: 0 4px 16px rgba(21,102,89,.1),  0 2px 6px rgba(21,102,89,.06);
    --shadow-lg: 0 12px 40px rgba(21,102,89,.13), 0 4px 12px rgba(21,102,89,.08);
    --radius: 14px;
    --radius-sm: 8px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--neutral-900) !important;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(150deg, var(--mint-50) 0%, #eaf9f7 40%, var(--sky-100) 100%) !important;
}

/* ── Main container ── */
.block-container {
    max-width: 1080px !important;
    padding: 2rem 2rem 3rem !important;
}

/* ── Headings ── */
h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: var(--teal-700) !important;
    margin-bottom: 0.25rem !important;
}
h2 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: var(--teal-700) !important;
    margin: 1.5rem 0 1rem 0 !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid var(--mint-200) !important;
}
h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--neutral-700) !important;
}

/* ── Labels ── */
label, .stSelectbox label, .stSlider label {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--neutral-700) !important;
    margin-bottom: 4px !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: white !important;
    border: 1.5px solid var(--mint-200) !important;
    border-radius: var(--radius-sm) !important;
    padding: 4px 8px !important;
}
.stSelectbox > div > div:hover {
    border-color: var(--mint-400) !important;
}

/* ── Slider ── */
.stSlider > div {
    padding: 10px 0 !important;
}
div[data-baseweb="slider"] > div:first-child > div {
    background: var(--mint-200) !important;
}
div[data-baseweb="slider"] [role="slider"] {
    background: white !important;
    border: 2px solid var(--mint-500) !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--mint-500) 0%, var(--sky-500) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: transform 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
}

/* ── Alert boxes ── */
.stAlert {
    border-radius: var(--radius) !important;
    border-left: 4px solid !important;
    padding: 1rem !important;
}

/* ── Card styling ── */
.css-1r6slb0, .stMarkdown {
    background: rgba(255,255,255,0.7);
    border-radius: var(--radius);
    padding: 0.5rem;
}

/* ── Divider ── */
hr {
    margin: 1rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model assets ───────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("churn_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        features = pickle.load(open("features.pkl", "rb"))
        return model, scaler, features
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found. Please ensure 'churn_model.pkl', 'scaler.pkl', and 'features.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model files: {str(e)}")
        st.stop()

model, scaler, features = load_assets()

# ── Hero header ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 10])
with col1:
    st.markdown("## 📡")
with col2:
    st.markdown("# Customer Churn Predictor")
st.caption("Fill in customer details below to predict the likelihood of churn")

# ── Helper maps ────────────────────────────────────────────────────────────────
BINARY = {0: "No", 1: "Yes"}
GENDER_MAP = {0: "Female", 1: "Male"}
LINES_MAP = {0: "No", 1: "Yes", 2: "No Phone Service"}
INET_MAP = {0: "DSL", 1: "Fiber Optic", 2: "No"}
SVC_MAP = {0: "No", 1: "Yes", 2: "No Internet Service"}
CONTRACT_MAP = {0: "Month-to-Month", 1: "One Year", 2: "Two Year"}
PAYMENT_MAP = {0: "Bank Transfer", 1: "Credit Card", 2: "Electronic Check", 3: "Mailed Check"}

def labeled_select(label, options_map, key):
    """Selectbox that shows human-readable labels, returns integer value."""
    labels = list(options_map.values())
    values = list(options_map.keys())
    chosen_label = st.selectbox(label, labels, key=key)
    return values[labels.index(chosen_label)]

# ── Form layout ────────────────────────────────────────────────────────────────
st.markdown("## 👤 Customer Profile")

col1, col2, col3, col4 = st.columns(4)
with col1:
    Gender = labeled_select("Gender", GENDER_MAP, "gender")
with col2:
    Senior_Citizen = labeled_select("Senior Citizen", BINARY, "senior")
with col3:
    Partner = labeled_select("Partner", BINARY, "partner")
with col4:
    Dependents = labeled_select("Dependents", BINARY, "dependents")

st.markdown("## 📱 Services")

col1, col2, col3 = st.columns(3)
with col1:
    Phone_Service = labeled_select("Phone Service", BINARY, "phone")
    Multiple_Lines = labeled_select("Multiple Lines", LINES_MAP, "lines")
with col2:
    Internet_Service = labeled_select("Internet Service", INET_MAP, "inet")
    Online_Security = labeled_select("Online Security", SVC_MAP, "sec")
with col3:
    Online_Backup = labeled_select("Online Backup", SVC_MAP, "backup")
    Device_Protection = labeled_select("Device Protection", SVC_MAP, "dev_prot")

col1, col2, col3 = st.columns(3)
with col1:
    Tech_Support = labeled_select("Tech Support", SVC_MAP, "tech")
with col2:
    Streaming_TV = labeled_select("Streaming TV", SVC_MAP, "stv")
with col3:
    Streaming_Movies = labeled_select("Streaming Movies", SVC_MAP, "smov")

st.markdown("## 💳 Billing & Contract")

col1, col2, col3 = st.columns(3)
with col1:
    Contract = labeled_select("Contract Type", CONTRACT_MAP, "contract")
with col2:
    Paperless_Billing = labeled_select("Paperless Billing", BINARY, "paperless")
with col3:
    Payment_Method = labeled_select("Payment Method", PAYMENT_MAP, "payment")

st.markdown("## 💰 Financials")

col1, col2, col3, col4 = st.columns(4)
with col1:
    Tenure_Months = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12, step=1)
with col2:
    Monthly_Charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0, step=5.0)
with col3:
    Total_Charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0, step=50.0)
with col4:
    CLTV = st.number_input("CLTV ($)", min_value=0.0, max_value=10000.0, value=2000.0, step=100.0)

st.markdown("---")

# ── Predict ────────────────────────────────────────────────────────────────────
# Prepare input data
input_dict = {
    'Gender': Gender,
    'Senior_Citizen': Senior_Citizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'Tenure_Months': Tenure_Months,
    'Phone_Service': Phone_Service,
    'Multiple_Lines': Multiple_Lines,
    'Internet_Service': Internet_Service,
    'Online_Security': Online_Security,
    'Online_Backup': Online_Backup,
    'Device_Protection': Device_Protection,
    'Tech_Support': Tech_Support,
    'Streaming_TV': Streaming_TV,
    'Streaming_Movies': Streaming_Movies,
    'Contract': Contract,
    'Paperless_Billing': Paperless_Billing,
    'Payment_Method': Payment_Method,
    'Monthly_Charges': Monthly_Charges,
    'Total_Charges': Total_Charges,
    'CLTV': CLTV,
}

# Create DataFrame with proper feature order
input_df = pd.DataFrame([input_dict])

# Ensure all expected features are present
for feature in features:
    if feature not in input_df.columns:
        input_df[feature] = 0

# Reorder columns to match training data
input_df = input_df[features]

# Scale the input
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"❌ Error scaling input data: {str(e)}")
    st.stop()

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button("🔍 Run Prediction", use_container_width=True)

if predict_clicked:
    try:
        # Get prediction and probability
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0]
        
        st.markdown("---")
        
        # Display result
        if prediction[0] == 1:
            st.error(
                f"**⚠️ High Churn Risk — {probability[1]*100:.1f}% Probability**\n\n"
                "This customer is likely to churn. Consider reaching out with a retention offer.",
                icon="🚨"
            )
        else:
            st.success(
                f"**✅ Low Churn Risk — {probability[0]*100:.1f}% Probability**\n\n"
                "This customer is likely to stay. Keep up the great service!",
                icon="🎉"
            )
        
        # Display key metrics summary
        st.markdown("### 📊 Key Metrics")
        metric_cols = st.columns(5)
        
        metrics = [
            ("Contract Type", CONTRACT_MAP[Contract]),
            ("Tenure", f"{Tenure_Months} months"),
            ("Monthly Charges", f"${Monthly_Charges:,.2f}"),
            ("Internet", INET_MAP[Internet_Service]),
            ("Payment Method", PAYMENT_MAP[Payment_Method].replace(" ", "\n")),
        ]
        
        for col, (label, value) in zip(metric_cols, metrics):
            col.metric(label=label, value=value)
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")