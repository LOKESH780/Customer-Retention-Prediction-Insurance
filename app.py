import streamlit as st
import joblib
import numpy as np
import pandas as pd
from login import login
from io import BytesIO
import base64

# === Background (optional) ===
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        h1, h2, h3 {{
            color: #ffffff;
        }}
        .stButton > button {{
            color: white;
            background-color: #0066cc;
            border-radius: 8px;
            padding: 0.5em 1em;
            min-width: 100px;
            text-align: center;
            font-weight: bold;
        }}
        </style>
    """, unsafe_allow_html=True)

add_bg_from_local("wallpaper.png")

# === Login Logic ===
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# === Logout Button ===
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

# === Load Model and Scaler ===
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("📈 Customer Retention Ratio Predictor")

input_method = st.radio("Select Input Method:", ["Manual Input", "Upload CSV File"])

feature_cols = [
    'AGENCY_APPOINTMENT_YEAR',
    'WRTN_PREM_AMT',
    'NB_WRTN_PREM_AMT',
    'POLY_INFORCE_QTY',
    'PREV_POLY_INFORCE_QTY',
    'LOSS_RATIO',
    'GROWTH_RATE_3YR',
    'ACTIVE_PRODUCERS'
]

# === Manual Input Mode ===
if input_method == "Manual Input":
    st.subheader("📝 Enter values manually")
    col1, col2 = st.columns(2)
    with col1:
        appoint_year = st.number_input("📅 Agency Appointment Year", min_value=1950, max_value=2050)
        wrtn_prem_amt = st.number_input("💵 Written Premium Amount", min_value=0.0)
        nb_wrtn_prem_amt = st.number_input("🆕 New Business Premium Amount", min_value=0.0)
        poly_qty = st.number_input("📄 Policies Inforce Quantity", min_value=0.0)
    with col2:
        prev_poly_qty = st.number_input("📄 Previous Policies Inforce Quantity", min_value=0.0)
        loss_ratio = st.number_input("📉 Loss Ratio", min_value=0.0, max_value=1000.0)
        growth_rate = st.number_input("📈 Growth Rate (3-Year)", min_value=-1.0)
        active_producers = st.number_input("👥 Active Producers", min_value=0.0)

    if st.button("Predict 🔮"):
        user_input = [[
            appoint_year,
            wrtn_prem_amt,
            nb_wrtn_prem_amt,
            poly_qty,
            prev_poly_qty,
            loss_ratio,
            growth_rate,
            active_producers
        ]]
        input_df = pd.DataFrame(user_input, columns=feature_cols)
        input_scaled = scaler.transform(input_df.fillna(0))
        prediction = model.predict(input_scaled)[0]
        st.success(f"🔐 Predicted Retention Ratio: {prediction:.2f}")

# === CSV Upload Mode ===
elif input_method == "Upload CSV File":
    st.subheader("📁 Upload CSV for batch prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("✅ File Uploaded Successfully!")

            if not all(col in df.columns for col in feature_cols):
                st.error(f"❌ CSV must contain the following columns: {', '.join(feature_cols)}")
            else:
                X = df[feature_cols].fillna(0)
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                df["Predicted Retention Ratio"] = predictions

                # === KPIs ===
                total_records = len(df)
                avg_retention = df["Predicted Retention Ratio"].mean()
                min_retention = df["Predicted Retention Ratio"].min()
                max_retention = df["Predicted Retention Ratio"].max()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Records", total_records)
                    st.metric("Average Retention Ratio", f"{avg_retention:.2f}")
                with col2:
                    st.metric("Min Retention", f"{min_retention:.2f}")
                    st.metric("Max Retention", f"{max_retention:.2f}")

                st.dataframe(df)

                csv_buffer = BytesIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv_data,
                    file_name="predicted_retention_ratios.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Something went wrong while reading the file: {e}")
