import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.markdown("Predict whether a telecom customer is likely to leave.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.radio("Senior Citizen?", ["No", "Yes"], horizontal=True)
    Partner = st.radio("Has Partner?", ["No", "Yes"], horizontal=True)
    Dependents = st.radio("Has Dependents?", ["No", "Yes"], horizontal=True)
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=20.0, max_value=120.0, value=65.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=800.0)

with col2:
    PhoneService = st.radio("Phone Service?", ["No", "Yes"], horizontal=True)
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.radio("Paperless Billing?", ["No", "Yes"], horizontal=True)
    PaymentMethod = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])

st.divider()

if st.button("Predict Churn", use_container_width=True):

    input_dict = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if SeniorCitizen == "Yes" else 0,
        'Partner': 1 if Partner == "Yes" else 0,
        'Dependents': 1 if Dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if PhoneService == "Yes" else 0,
        'PaperlessBilling': 1 if PaperlessBilling == "Yes" else 0,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'MultipleLines_No phone service': 1 if MultipleLines == "No phone service" else 0,
        'MultipleLines_Yes': 1 if MultipleLines == "Yes" else 0,
        'InternetService_Fiber optic': 1 if InternetService == "Fiber optic" else 0,
        'InternetService_No': 1 if InternetService == "No" else 0,
        'OnlineSecurity_No internet service': 1 if OnlineSecurity == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if OnlineSecurity == "Yes" else 0,
        'OnlineBackup_No internet service': 1 if OnlineBackup == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if OnlineBackup == "Yes" else 0,
        'DeviceProtection_No internet service': 1 if DeviceProtection == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if DeviceProtection == "Yes" else 0,
        'TechSupport_No internet service': 1 if TechSupport == "No internet service" else 0,
        'TechSupport_Yes': 1 if TechSupport == "Yes" else 0,
        'StreamingTV_No internet service': 1 if StreamingTV == "No internet service" else 0,
        'StreamingTV_Yes': 1 if StreamingTV == "Yes" else 0,
        'StreamingMovies_No internet service': 1 if StreamingMovies == "No internet service" else 0,
        'StreamingMovies_Yes': 1 if StreamingMovies == "Yes" else 0,
        'Contract_One year': 1 if Contract == "One year" else 0,
        'Contract_Two year': 1 if Contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if PaymentMethod == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if PaymentMethod == "Mailed check" else 0,
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    if prediction == 1:
        st.error(f"High Churn Risk - {probability:.0%} probability")
        st.markdown("""
        **Recommended Actions:**
        - Offer a loyalty discount or free upgrade
        - Assign a dedicated support agent
        - Propose switching to an annual contract
        """)
    else:
        st.success(f"Low Churn Risk - {probability:.0%} probability")
        st.markdown("""
        **Customer looks stable. Consider:**
        - Upsell premium features
        - Enrol in loyalty rewards program
        """)

    st.markdown("#### Churn Probability")
    st.progress(float(probability))
    st.caption(f"{probability:.1%} likelihood of churning")
