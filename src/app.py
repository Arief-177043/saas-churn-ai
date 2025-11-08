# ==============================================
# 🚀 SAAS Churn Predictor + Explainer Agent (Advanced UI + AI Report + Bulk Email)
# ==============================================

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from feature_extractor import aggregate_features
from predictor import predict
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
from fpdf import FPDF  # ✅ PDF Export Feature
import smtplib
from email.mime.text import MIMEText

# -----------------------------
# 🌟 Page Setup
# -----------------------------
st.set_page_config(page_title="SAAS Churn Predictor", layout="wide", page_icon="🤖")

# -----------------------------
# 🔑 GEMINI API CONFIG
# -----------------------------
import google.generativeai as genai
GEMINI_API_KEY = "AIzaSyDCR8IG_Jb5OS3_OAUORvK5G_U22uL-1yk"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# 🎞 Lottie Loader
# -----------------------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# -----------------------------
# 🌈 Keep Original Background
# -----------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
h1, h2, h3 {
    text-shadow: 0px 0px 20px rgba(0,255,255,0.3);
    animation: fadeIn 1.5s ease-in-out;
}
@keyframes fadeIn {
    0% {opacity: 0; transform: translateY(-10px);}
    100% {opacity: 1; transform: translateY(0);}
}
.footer {
    text-align: center;
    padding: 15px;
    color: #9ae3ff;
    margin-top: 30px;
    border-top: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 🧭 NAVBAR
# -----------------------------
selected = option_menu(
    menu_title=None,
    options=["🏠 Home", "📈 Prediction", "💬 Assistant", "👥 Team"],
    icons=["house", "bar-chart", "robot", "people"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#111827"},
        "icon": {"color": "#00E0FF", "font-size": "18px"},
        "nav-link": {"color": "#E5E7EB", "font-size": "16px"},
        "nav-link-selected": {"background-color": "#2563EB"},
    },
)

# -----------------------------
# 🏠 HOME
# -----------------------------
if selected == "🏠 Home":
    st.markdown("<h1 style='text-align:center;'>🚀 SAAS Churn Predictor + Explainer Agent</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#D1FAE5;'>Predict churn, explain insights, and retain customers smartly.</p>", unsafe_allow_html=True)
    lottie_home = load_lottie("https://assets7.lottiefiles.com/packages/lf20_ydo1amjm.json")
    st_lottie(lottie_home, height=350, key="home")
    st.markdown("<div style='text-align:center; color:#9AE3FF;'>Upload customer logs → Analyze engagement → Predict churn → Get AI insights.</div>", unsafe_allow_html=True)

# -----------------------------
# 📈 PREDICTION
# -----------------------------
elif selected == "📈 Prediction":
    st.markdown("<h2 style='text-align:center; color:#00E0FF;'>📈 Churn Prediction Dashboard</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📂 Upload your customer_data.csv", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Data loaded successfully!")
        st.dataframe(df.head(), use_container_width=True)

        with st.spinner("🔄 Extracting behavioral features..."):
            features = aggregate_features(df)
        with st.spinner("🤖 Predicting churn probabilities..."):
            probs = predict(features.drop(columns=["customer_id"]))
            features["churn_probability"] = probs

        def categorize(p):
            if p < 0.3: return "Low Risk"
            elif p < 0.6: return "Medium Risk"
            else: return "High Risk"

        features["risk_level"] = features["churn_probability"].apply(categorize)
        avg_churn = features["churn_probability"].mean() * 100
        total_customers = len(features)
        high_risk = (features["risk_level"] == "High Risk").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("👥 Total Customers", total_customers)
        col2.metric("🔥 High Risk Customers", high_risk)
        col3.metric("📊 Avg Churn Probability", f"{avg_churn:.2f}%")

        st.dataframe(features[["customer_id", "churn_probability", "risk_level"]], use_container_width=True)
        st.session_state["features"] = features
    else:
        st.info("⬆️ Upload your CSV to start predictions.")

# -----------------------------
# 💬 ASSISTANT + AI Report + Bulk Email
# -----------------------------
elif selected == "💬 Assistant":
    st.markdown("<h2 style='text-align:center; color:#00E0FF;'>💬 AI Churn Explainer Assistant</h2>", unsafe_allow_html=True)
    if "features" not in st.session_state:
        st.warning("⚠️ Please upload data on Prediction page first.")
    else:
        features = st.session_state["features"]
        cid = st.selectbox("Select Customer ID:", features["customer_id"].unique())
        row = features[features["customer_id"] == cid].iloc[0]
        prob, risk = row["churn_probability"], row["risk_level"]

        st.progress(float(prob))
        if risk == "High Risk":
            msg = f"⚠️ **High Churn Risk ({prob:.2f})** → Immediate retention required."
        elif risk == "Medium Risk":
            msg = f"🟡 **Moderate Risk ({prob:.2f})** → Send re-engagement offers."
        else:
            msg = f"🟢 **Low Risk ({prob:.2f})** → Customer is happy."
        st.info(msg)

        # 🔹 Ask AI
        st.markdown("### 🧠 Ask AI Assistant for Explanation")
        user_prompt = st.text_area("💬 Example: Why is this customer likely to churn?")
        if st.button("Generate AI Insight"):
            with st.spinner("🤖 Generating AI Insight..."):
                try:
                    model = genai.GenerativeModel("gemini-2.0-flash-lite")
                    full_prompt = f"""
                    You are an AI churn analysis assistant. Analyze:
                    {row.to_dict()}
                    Question: {user_prompt}
                    Give a concise business insight.
                    """
                    response = model.generate_content(full_prompt)
                    insight = response.text
                    st.success("✅ AI insight generated!")
                    st.markdown(f"### 🧩 Insight:\n\n{insight}")
                    # Strategy
                    strategy = model.generate_content(f"Suggest a personalized retention strategy: {row.to_dict()}").text
                    st.markdown(f"### 💡 Retention Strategy:\n{strategy}")
                    # PDF Export
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="SAAS Churn AI Report", ln=True, align="C")
                    pdf.multi_cell(0, 10, txt=f"Customer ID: {cid}")
                    pdf.multi_cell(0, 10, txt=f"Churn Probability: {prob:.2f}")
                    pdf.multi_cell(0, 10, txt=f"Risk Level: {risk}")
                    pdf.multi_cell(0, 10, txt=f"AI Insight:\n{insight}")
                    pdf.multi_cell(0, 10, txt=f"Retention Strategy:\n{strategy}")
                    pdf_path = f"models/churn_report_{cid}.pdf"
                    pdf.output(pdf_path)
                    with open(pdf_path, "rb") as f:
                        st.download_button("📄 Download AI Report", f, file_name=f"churn_report_{cid}.pdf")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # 🚀 Bulk Retention Campaign
        st.markdown("---")
        st.markdown("## 📢 Smart Retention Campaign (Bulk Email Automation)")
        high_risk_customers = features[features["risk_level"] == "High Risk"]
        if len(high_risk_customers) == 0:
            st.info("✅ No high-risk customers currently.")
        else:
            st.markdown(f"**{len(high_risk_customers)} High-Risk Customers Detected.**")
            st.dataframe(high_risk_customers[["customer_id", "churn_probability"]])

            sender_email = st.text_input("📨 Sender Email (your Gmail)", key="bulk_sender")
            sender_pass = st.text_input("🔑 App Password (for Gmail)", type="password")
            if st.button("🤖 Generate Retention Emails"):
                with st.spinner("✍️ AI drafting personalized retention emails..."):
                    try:
                        model = genai.GenerativeModel("gemini-2.0-flash-lite")
                        emails = []
                        for _, row in high_risk_customers.iterrows():
                            prompt = f"Write a short retention email for customer {row['customer_id']} (churn prob: {row['churn_probability']:.2f})."
                            resp = model.generate_content(prompt)
                            emails.append({"id": row["customer_id"], "email": resp.text})
                        st.session_state["emails"] = emails
                        st.success("✅ AI generated all emails!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

            if "emails" in st.session_state:
                for e in st.session_state["emails"]:
                    st.markdown(f"**Customer {e['id']}**")
                    st.text_area("", e["email"], height=150, key=f"mail_{e['id']}")
                if st.button("🚀 Send All Emails"):
                    try:
                        with smtplib.SMTP("smtp.gmail.com", 587) as server:
                            server.starttls()
                            server.login(sender_email, sender_pass)
                            for e in st.session_state["emails"]:
                                msg = MIMEText(e["email"])
                                msg["Subject"] = "We Value You – Let’s Stay Connected 💙"
                                msg["From"] = sender_email
                                msg["To"] = f"customer_{e['id']}@example.com"  # Demo
                                server.send_message(msg)
                        st.success("✅ All retention emails sent successfully!")
                    except Exception as e:
                        st.error(f"❌ Email Sending Failed: {str(e)}")

# -----------------------------
# 👥 TEAM
# -----------------------------
elif selected == "👥 Team":
    st.markdown("<h2 style='text-align:center; color:#00E0FF;'>👥 Meet the Team</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#9AE3FF;'>
        <b>Team VisionX AI</b><br>
        <p>Y Mohammed Arief — ML Developer | Team Lead</p>
        <p>Bhabitha — Data Engineer | Feature Extraction</p>
        <p>Rethaka — UI/UX Developer | Streamlit Visualization</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# 🆘 ABOUT
# -----------------------------
with st.expander("ℹ️ About Project"):
    st.markdown("""
    ### 🎯 Overview
    **SaaS Churn Predictor + Explainer Agent** predicts churn, explains why, and automates retention.
    - Real-time churn prediction (**LightGBM**)
    - Explainable AI (**SHAP**)
    - AI customer insights (**Gemini**)
    - 📄 AI PDF Reports
    - ✉️ Smart Retention Campaign (Bulk AI Email)
    """)

st.markdown("""
<div class="footer">
💡 Built by <b>Team VisionX AI</b> | Powered by LightGBM + Gemini + Streamlit
</div>
""", unsafe_allow_html=True)
