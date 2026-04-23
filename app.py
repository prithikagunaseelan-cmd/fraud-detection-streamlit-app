import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import uuid

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import plotly.graph_objects as go

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Risk Analytics - Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CSS
# -------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        max-width: 1400px;
    }

    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #15284b;
        margin-bottom: 0.2rem;
        line-height: 1.2;
        word-break: break-word;
    }

    .app-subtitle {
        font-size: 1.02rem;
        color: #5f6f8a;
        margin-bottom: 1rem;
        line-height: 1.6;
        word-break: break-word;
    }

    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        text-align: center;
        min-height: 105px;
    }

    .metric-label {
        font-size: 0.78rem;
        letter-spacing: 1px;
        color: #8a96ad;
        font-weight: 700;
        text-transform: uppercase;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #2962ff;
        margin-top: 8px;
    }

    .panel-card {
        background: white;
        border-radius: 26px;
        padding: 24px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.07);
        overflow: visible;
    }

    .dark-panel {
        background: #172b4d;
        color: white;
        border-radius: 26px;
        padding: 24px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.09);
        overflow: visible;
    }

    .warning-pill {
        display: inline-block;
        background: #fff3d6;
        color: #b56a00;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 999px;
        font-size: 0.9rem;
        margin-bottom: 10px;
        line-height: 1.3;
        word-break: break-word;
    }

    .hero-heading {
        font-size: 2rem;
        font-weight: 800;
        color: #15284b;
        line-height: 1.2;
        margin-bottom: 10px;
        word-break: break-word;
    }

    .hero-text {
        color: #66758f;
        font-size: 1.03rem;
        line-height: 1.7;
        word-break: break-word;
    }

    .dark-title {
        font-size: 1rem;
        letter-spacing: 2px;
        font-weight: 800;
        color: #94a3c3;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    .report-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        padding: 12px 0;
        border-bottom: 1px solid rgba(255,255,255,0.10);
    }

    .report-key {
        color: #b9c6df;
        font-weight: 700;
        font-size: 1rem;
        word-break: break-word;
    }

    .report-value {
        color: #4da3ff;
        font-weight: 800;
        font-size: 1.35rem;
        text-align: right;
        word-break: break-word;
    }

    .small-card {
        background: rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 16px;
        margin-top: 16px;
    }

    .small-title {
        color: #b9c6df;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .section-title {
        font-size: 1.12rem;
        font-weight: 800;
        color: #15284b;
        margin-top: 14px;
        margin-bottom: 8px;
        word-break: break-word;
    }

    .audit-box {
        background: #f1f5fb;
        border-radius: 18px;
        padding: 16px;
        color: #24344f;
        line-height: 1.8;
        margin-top: 10px;
        word-break: break-word;
    }

    .engine-box {
        background: #f4f7fb;
        border: 1px solid #e4e9f3;
        border-radius: 16px;
        padding: 16px;
        margin-top: 18px;
        color: #44536d;
        font-family: monospace;
        font-size: 0.92rem;
        line-height: 1.7;
    }

    div[data-testid="stMetric"] {
        background: transparent;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA + MODEL
# -------------------------------------------------
from sklearn.datasets import make_classification

@st.cache_data
def load_data():
    X, y = make_classification(
        n_samples=5000,
        n_features=30,
        n_informative=10,
        n_redundant=10,
        weights=[0.98],
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 31)])
    df["Amount"] = np.random.uniform(10, 50000, size=len(df))
    df["Class"] = y

    return df

@st.cache_resource
def prepare_model():
    df = load_data().copy()

    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "n_samples": len(df)
    }

    return model, scaler, metrics

model, scaler, metrics = prepare_model()

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def risk_level(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.70:
        return "Moderate Risk"
    elif prob < 0.90:
        return "High Risk"
    return "Critical Risk"

def decision_action(prob):
    if prob < 0.30:
        return "APPROVE TRANSACTION"
    elif prob < 0.70:
        return "OTP / MANUAL VERIFICATION"
    elif prob < 0.90:
        return "HOLD & REVIEW"
    return "BLOCK TRANSACTION"

def warning_text(prob):
    if prob < 0.30:
        return "No unusual fraud behavior detected"
    elif prob < 0.70:
        return "Unusual Behavioral Patterns Detected"
    elif prob < 0.90:
        return "High Fraud Risk Pattern Detected"
    return "Critical Fraud Pattern Detected"

def recommendation_text(prob):
    if prob < 0.30:
        return "Transaction appears safe. Continue normal monitoring."
    elif prob < 0.70:
        return "Manual verification recommended. Trigger OTP and customer confirmation."
    elif prob < 0.90:
        return "Place transaction on hold and escalate to fraud analyst."
    return "Block transaction immediately, escalate alert, and notify fraud operations team."

def case_priority(prob):
    if prob < 0.30:
        return "Low"
    elif prob < 0.70:
        return "Medium"
    elif prob < 0.90:
        return "High"
    return "Critical"

def make_ticket():
    return "FRAUD-" + uuid.uuid4().hex[:8].upper()

def build_gauge(score_percent: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_percent,
        number={"font": {"size": 42, "color": "#d97706"}},
        title={"text": "RISK INDEX", "font": {"size": 16, "color": "#8a96ad"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 0},
            "bar": {"color": "#f59e0b", "thickness": 0.34},
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#ecfdf5"},
                {"range": [30, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#fee2e2"},
            ],
        }
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white"
    )
    return fig

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("## Risk Analytics")
    st.caption("FRAUD ANALYTICS ENGINE")

    st.markdown("### Customer Profile")
    customer_id = st.text_input("Customer ID", value="FBI-ACC-10021")
    customer_name = st.text_input("Customer Name", value="Federal Bank Client A")
    account_type = st.selectbox("Account Type", ["Savings", "Current", "Institutional"])
    customer_segment = st.selectbox("Customer Segment", ["Retail", "SME", "Corporate", "Institutional"])
    relationship_years = st.slider("Relationship with Bank (Years)", 0, 30, 5)
    account_age_months = st.slider("Account Age (Months)", 0, 240, 48)
    kyc_status = st.selectbox("KYC Status", ["Verified", "Pending", "High-Risk Review"])
    avg_monthly_amount = st.number_input("Average Monthly Transaction Amount", min_value=0.0, value=25000.0)

    st.markdown("### Transaction Parameters")
    scenario = st.selectbox(
        "Quick Scenario",
        ["Custom", "Normal Case", "Suspicious Case", "High Risk Case"]
    )

    if scenario == "Normal Case":
        default_amount = 3000.0
        default_hour = 14
        default_distance = 5.0
        default_international = "No"
        default_new_device = "No"
        default_merchant = "Low"
        default_txn24 = 2
        default_failed = 0
    elif scenario == "Suspicious Case":
        default_amount = 25000.0
        default_hour = 23
        default_distance = 120.0
        default_international = "Yes"
        default_new_device = "Yes"
        default_merchant = "Medium"
        default_txn24 = 12
        default_failed = 2
    elif scenario == "High Risk Case":
        default_amount = 100000.0
        default_hour = 2
        default_distance = 230.0
        default_international = "Yes"
        default_new_device = "Yes"
        default_merchant = "High"
        default_txn24 = 18
        default_failed = 4
    else:
        default_amount = 10000.0
        default_hour = 2
        default_distance = 200.0
        default_international = "No"
        default_new_device = "No"
        default_merchant = "Low"
        default_txn24 = 8
        default_failed = 1

    amount_val = st.number_input("Amount (INR)", min_value=0.0, value=float(default_amount))
    hour_val = st.slider("Hour of Day", 0, 23, int(default_hour))
    distance_val = st.number_input("Distance (KM)", min_value=0.0, value=float(default_distance))
    international_status = st.radio(
        "International Status",
        ["No", "Yes"],
        horizontal=True,
        index=0 if default_international == "No" else 1
    )
    new_device = st.radio(
        "New Device",
        ["No", "Yes"],
        horizontal=True,
        index=0 if default_new_device == "No" else 1
    )
    merchant_risk = st.selectbox(
        "Merchant Risk",
        ["Low", "Medium", "High"],
        index=["Low", "Medium", "High"].index(default_merchant)
    )
    txn_count_24h = st.slider("Transactions (24h)", 0, 50, int(default_txn24))
    failed_attempts = st.slider("Failed Attempts", 0, 10, int(default_failed))

    recalc = st.button("🔄 Re-Calculate Risk", use_container_width=True)

    st.markdown("""
    <div class="engine-box">
    <b>SYSTEM ENGINE</b><br>
    MODEL: LOGISTIC_REG_V4.2<br>
    LAST TRAINED: 2026-04-22<br>
    ENV: PRODUCTION_STABLE
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# HYBRID SCORE
# -------------------------------------------------
input_df = pd.DataFrame(np.zeros((1, 31)), columns=[f"V{i}" for i in range(1, 31)] + ["Amount"])

input_df["V1"] = hour_val
input_df["Amount"] = scaler.transform(pd.DataFrame({"Amount": [amount_val]}))[0][0]

base_prob = model.predict_proba(input_df)[0][1]

behavioral_risk = 0.0
risk_reasons = []

if amount_val > 50000:
    behavioral_risk += 0.12
    risk_reasons.append("High transaction amount")
if amount_val > 100000:
    behavioral_risk += 0.12
    risk_reasons.append("Very high transaction amount")
if amount_val > avg_monthly_amount * 2:
    behavioral_risk += 0.10
    risk_reasons.append("Amount significantly above customer average")
if hour_val < 5 or hour_val > 22:
    behavioral_risk += 0.10
    risk_reasons.append("Transaction in unusual time window")
if distance_val > 100:
    behavioral_risk += 0.08
    risk_reasons.append("Far from usual location")
if distance_val > 200:
    behavioral_risk += 0.10
    risk_reasons.append("Extreme distance anomaly")
if international_status == "Yes":
    behavioral_risk += 0.18
    risk_reasons.append("International transaction")
if txn_count_24h > 10:
    behavioral_risk += 0.08
    risk_reasons.append("High daily transaction frequency")
if txn_count_24h > 20:
    behavioral_risk += 0.10
    risk_reasons.append("Very high transaction burst")
if failed_attempts > 2:
    behavioral_risk += 0.12
    risk_reasons.append("Multiple failed attempts")
if new_device == "Yes":
    behavioral_risk += 0.12
    risk_reasons.append("New device detected")
if merchant_risk == "Medium":
    behavioral_risk += 0.08
    risk_reasons.append("Medium-risk merchant")
elif merchant_risk == "High":
    behavioral_risk += 0.16
    risk_reasons.append("High-risk merchant")

if account_age_months < 6:
    behavioral_risk += 0.08
    risk_reasons.append("Newly opened account")
if relationship_years < 1:
    behavioral_risk += 0.06
    risk_reasons.append("Short banking relationship")
if kyc_status == "Pending":
    behavioral_risk += 0.10
    risk_reasons.append("Incomplete KYC")
elif kyc_status == "High-Risk Review":
    behavioral_risk += 0.16
    risk_reasons.append("Customer under high-risk KYC review")
if customer_segment in ["Corporate", "Institutional"] and amount_val > 50000:
    behavioral_risk += 0.05
    risk_reasons.append("Large-value institutional transaction")

if not risk_reasons:
    risk_reasons.append("No major behavioral anomaly detected")

final_prob = min(1.0, base_prob + behavioral_risk)
risk_percent = round(final_prob * 100, 1)
risk_label = risk_level(final_prob)
decision = decision_action(final_prob)
summary_text = warning_text(final_prob)
recommendation = recommendation_text(final_prob)
priority = case_priority(final_prob)
ticket_id = make_ticket()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
left_top, card1, card2 = st.columns([5, 1.2, 1.2])

with left_top:
    st.markdown('<div class="app-title">RISK Analysis Report</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Real-time fraud scoring for banking transactions using a hybrid machine learning and behavioral risk framework.</div>',
        unsafe_allow_html=True
    )

with card1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value">{metrics['accuracy']*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with card2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Recall (Fraud)</div>
        <div class="metric-value">{metrics['recall']*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------
col_left, col_right = st.columns([2.2, 1], gap="large")

with col_left:
    with st.container():
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        gauge_col, text_col = st.columns([1.0, 1.25], gap="medium")

        with gauge_col:
            st.plotly_chart(build_gauge(risk_percent), use_container_width=True)

        with text_col:
            pill_text = (
                "LOW RISK STATUS" if final_prob < 0.30
                else "MODERATE WARNING" if final_prob < 0.70
                else "HIGH RISK ALERT"
            )
            st.markdown(f'<div class="warning-pill">⚠ {pill_text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="hero-heading">{summary_text}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="hero-text">{recommendation}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Live Transaction Decision</div>', unsafe_allow_html=True)
        if final_prob < 0.30:
            st.success(f"✅ Decision: {decision}")
        elif final_prob < 0.70:
            st.warning(f"🔐 Decision: {decision}")
        elif final_prob < 0.90:
            st.warning(f"⏸ Decision: {decision}")
        else:
            st.error(f"🚫 Decision: {decision}")

        st.markdown('<div class="section-title">Real-Time Fraud Alert Simulation</div>', unsafe_allow_html=True)
        if final_prob < 0.30:
            st.info("No fraud alert generated. Transaction stored for routine monitoring.")
        elif final_prob < 0.70:
            st.warning("📩 Medium-priority alert sent for manual verification and OTP workflow.")
        elif final_prob < 0.90:
            st.warning("🚨 High-priority alert sent to fraud monitoring queue.")
        else:
            st.error("🚨 Critical alert sent to fraud operations team and customer notification triggered.")

        st.markdown('<div class="section-title">Risk Factors Detected</div>', unsafe_allow_html=True)
        for reason in risk_reasons:
            st.write(f"- {reason}")

        st.markdown('<div class="section-title">Customer Profile Summary</div>', unsafe_allow_html=True)
        st.write(f"**Customer ID:** {customer_id}")
        st.write(f"**Customer Name:** {customer_name}")
        st.write(f"**Customer Segment:** {customer_segment}")
        st.write(f"**Account Type:** {account_type}")
        st.write(f"**Relationship with Bank (Years):** {relationship_years}")
        st.write(f"**Account Age (Months):** {account_age_months}")
        st.write(f"**KYC Status:** {kyc_status}")
        st.write(f"**Avg Monthly Amount:** ₹{avg_monthly_amount:,.2f}")

        st.markdown('<div class="section-title">Case Log / Audit Trail</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="audit-box">
            <b>Timestamp:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            <b>Incident Ticket:</b> {ticket_id}<br>
            <b>Customer ID:</b> {customer_id}<br>
            <b>Customer Name:</b> {customer_name}<br>
            <b>Transaction Amount:</b> ₹{amount_val:,.2f}<br>
            <b>Hour of Day:</b> {hour_val}:00<br>
            <b>Distance:</b> {distance_val:.1f} KM<br>
            <b>Behavioral Risk Score:</b> {behavioral_risk:.3f}<br>
            <b>Final Fraud Probability:</b> {final_prob:.4f}<br>
            <b>Risk Level:</b> {risk_label}<br>
            <b>Decision Taken:</b> {decision}<br>
            <b>Case Priority:</b> {priority}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="dark-panel">', unsafe_allow_html=True)
    st.markdown('<div class="dark-title">Classification Report</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="report-row">
        <div class="report-key">Precision</div>
        <div class="report-value">{metrics['precision']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="report-row">
        <div class="report-key">F1-Score</div>
        <div class="report-value">{metrics['f1']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="report-row">
        <div class="report-key">Sensitivity</div>
        <div class="report-value">{metrics['recall']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="small-card">', unsafe_allow_html=True)
    st.markdown('<div class="small-title">Unique Features Enabled</div>', unsafe_allow_html=True)
    st.write("- Hybrid ML + behavioral scoring")
    st.write("- Customer profile summary")
    st.write("- Live transaction decision")
    st.write("- Fraud alert simulation")
    st.write("- Risk factor explanation")
    st.write("- Incident ticket + audit trail")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="small-card">', unsafe_allow_html=True)
    st.markdown('<div class="small-title">Case Priority</div>', unsafe_allow_html=True)
    if priority == "Low":
        st.success(priority)
    elif priority == "Medium":
        st.warning(priority)
    elif priority == "High":
        st.warning(priority)
    else:
        st.error(priority)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="small-card">', unsafe_allow_html=True)
    st.markdown('<div class="small-title">N-Samples</div>', unsafe_allow_html=True)
    st.write(f"{metrics['n_samples']:,}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if recalc:
    st.toast("Risk recalculated successfully.")