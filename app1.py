import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# CUSTOM STYLING (FULL WIDTH + BABY PINK THEME)
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fff0f5; /* baby pink */
    }
    .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="PCOS Early Risk Screening",
    page_icon="🩺",
    layout="centered"
)

# =========================================================
# LOAD MODEL & FEATURE SCHEMA
# =========================================================
model = joblib.load("pcos_xgb_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================================================
# HEADER
# =========================================================
st.title("🩺 PCOS Early Risk Screening Tool")

st.markdown("""
This application estimates **early PCOS risk probability**
using clinical and lifestyle indicators.

⚠️ *This is a screening support tool — not a diagnostic system.*
""")

st.divider()

# =========================================================
# FILE UPLOAD
# =========================================================
st.subheader("📤 Upload Patient Clinical Data")

uploaded_file = st.file_uploader(
    "Upload CSV (same format as streamlit_test_input.csv)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # ===============================
        # READ DATA
        # ===============================
        data = pd.read_csv(uploaded_file)

        st.subheader("🧾 Uploaded Patient Data")
        st.dataframe(data, use_container_width=True)

        # ===============================
        # NUMERIC SAFETY
        # ===============================
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data.fillna(0, inplace=True)

        # ===============================
        # 🚨 CRITICAL FIX: FEATURE ALIGNMENT
        # ===============================
        data = data.reindex(columns=feature_names, fill_value=0)

        # ===============================
# SYMPTOM CHECK (FOR EXPLANATION)
# ===============================
        symptom_features = [
            "hair growth(Y/N)",
            "Hair loss(Y/N)",
            "Pimples(Y/N)",
            "Skin darkening (Y/N)",
            "Weight gain(Y/N)",
            "Fast food (Y/N)",
            "Reg.Exercise(Y/N)"
        ]

        present_symptoms = []

        for feat in symptom_features:
            if feat in data.columns and data[feat].iloc[0] == 1:
                present_symptoms.append(feat.replace("(Y/N)", "").strip())


        # ===============================
        # PREDICTION (NO SCALER ❌)
        # ===============================
        risk_prob = model.predict_proba(data)[0][1]

        # ===============================
        # RISK BUCKETS
        # ===============================
        if risk_prob >= 0.71:
            risk_level = "High Risk"
            emoji = "🔴"
        elif risk_prob >= 0.4 :
            risk_level = "Moderate Risk"
            emoji = "🟠"
        else:
            risk_level = "Low Risk"
            emoji = "🟢"

        st.divider()

        # ===============================
        # RESULTS
        # ===============================
        st.subheader("🔍 Risk Assessment Result")

        col1, col2 = st.columns(2)
        col1.metric("PCOS Risk Probability", f"{risk_prob:.2f}")
        col2.metric("Risk Category", f"{emoji} {risk_level}")

        st.divider()

        # ===============================
        # CLINICAL SUGGESTIONS
        # ===============================
        st.subheader("📌 Screening & Lifestyle Suggestions")

        if risk_level == "High Risk":
            st.error("⚠️ **High Risk Indicators Detected**")

            if present_symptoms:
                st.markdown("**Key contributing factors to monitor:**")
                for s in present_symptoms:
                    st.write(f"• {s}")
                st.markdown("""
        **Recommended Actions**
        
        • Consult a gynecologist / endocrinologist  
        • Track menstrual irregularities  
        • Improve diet and physical activity  
        • Follow up with routine blood tests  
                """)
            else:
                st.write("Multiple clinical and metabolic indicators contributed to this risk.")

                


        elif risk_level == "Moderate Risk":
            st.warning("🟠 **Moderate Risk Indicators Detected**")

            if present_symptoms:
                st.markdown("**Symptoms that may need attention:**")
                for s in present_symptoms:
                    st.write(f"• {s}")
                st.markdown("""
            **Recommended Actions**
            • Monitor symptoms regularly  
            • Maintain healthy diet  
            • Increase physical activity  
            """)
            else:
                st.write("Lifestyle and metabolic patterns suggest moderate risk.")

           

        else:

            st.markdown("<span style='font-size:28px; color:green;'>All is well! 😊</span>", unsafe_allow_html=True)


            st.markdown("""
            Your indicators suggest **low PCOS risk at this time**.

            ✔ Maintain current lifestyle  
            ✔ Continue routine health monitoring  
            ✔ Stay physically active  
            """)

        st.caption(
            "⚠️ This tool provides early screening support only and does not diagnose PCOS."
        )

    except Exception as e:
        st.error("❌ Error processing file. Please upload a valid CSV.")
        st.text(str(e))
