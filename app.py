import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page setup
st.set_page_config(page_title="Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.markdown("Predict whether a customer is likely to leave.")
st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges ($)", 
                                      min_value=20.0, 
                                      max_value=120.0, 
                                      value=65.0)
    TotalCharges = st.number_input("Total Charges ($)", 
                                    min_value=0.0, 
                                    max_value=9000.0, 
                                    value=800.0)

with col2:
    contract = st.selectbox("Contract Type", 
                             ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", 
                             ["DSL", "Fiber optic", "No"])
    senior = st.radio("Senior Citizen?", ["No", "Yes"], horizontal=True)

st.divider()

# Button
if st.button("🔍 Predict Churn", use_container_width=True):

    # Encode inputs
    contract_one_year = 1 if contract == "One year" else 0
    contract_two_year = 1 if contract == "Two year" else 0
    internet_fiber = 1 if internet == "Fiber optic" else 0
    internet_no = 1 if internet == "No" else 0
    senior_val = 1 if senior == "Yes" else 0

    # Create input array — order must match training
    input_data = np.array([[tenure, MonthlyCharges, TotalCharges,
                             contract_one_year, contract_two_year,
                             internet_fiber, internet_no, senior_val]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — {probability:.0%} probability")
        st.markdown("""
        **Recommended Actions:**
        - 🎁 Offer a loyalty discount
        - 📞 Assign dedicated support
        - 📋 Suggest switching to annual contract
        """)
    else:
        st.success(f"✅ Low Churn Risk — {probability:.0%} probability")
        st.markdown("""
        **Customer looks stable. Consider:**
        - 🌟 Upsell premium features
        - 📧 Enrol in loyalty rewards
        """)

    st.markdown("#### Churn Probability")
    st.progress(float(probability))
    st.caption(f"{probability:.1%} chance of churning")
```

> ⚠️ **Important note:** This app.py assumes specific features. After we deploy it, if you get a feature mismatch error, paste the column list from Step 1.3 here and I'll fix the code instantly for you.

---

### Step 3.3 — Save the file

1. Scroll down to **"Commit changes"**
2. Leave the message as default
3. Click green **"Commit changes"**

You're back on your repo. You should now see:
```
📄 README.md
📄 app.py          ← new!
📦 churn_model.pkl
📦 scaler.pkl
```

---

### Step 3.4 — Create requirements.txt

Repeat the same process:

1. Click **"Add file"** → **"Create new file"**
2. Name it exactly: `requirements.txt`
3. Paste this in the text area:
```
streamlit
scikit-learn
xgboost
joblib
pandas
numpy
```

4. Scroll down → **"Commit changes"** → green button

Your repo should now have 5 files:
```
📄 README.md
📄 app.py
📄 requirements.txt
📦 churn_model.pkl
📦 scaler.pkl
```

✅ **Stage 3 done.**

---

# STAGE 4 — Deploy on Streamlit Cloud

**Open [share.streamlit.io](https://share.streamlit.io) in your browser.**

---

### Step 4.1 — Sign in

1. Click **"Sign in with GitHub"**
2. It'll ask permission to connect to GitHub — click **"Authorize"**
3. You're now logged into Streamlit Cloud

---

### Step 4.2 — Create your app

1. Click the **"New app"** button
2. You'll see a form — fill it like this:
```
Repository:   YourGitHubUsername/churn-prediction
Branch:       main
Main file:    app.py
```

3. Click the green **"Deploy!"** button

---

### Step 4.3 — Wait 2–3 minutes

You'll see a screen that says "Your app is in the oven 🍕"

Streamlit is:
- Reading your requirements.txt
- Installing all libraries
- Loading your model
- Starting the app

When it's done, your browser will automatically show your **live app.**

---

## What You'll Have at the End
```
🔗 Your live app URL will look like:
https://yourusername-churn-prediction-app-xyz123.streamlit.app
```

This link:
- Works 24/7 permanently
- Anyone in the world can open it
- Put it on your resume, LinkedIn, and GitHub README

---

## If You See an Error (Most Common Fix)

The most likely error is a **feature mismatch** — it'll look like:
```
ValueError: X has 7 features but model expects 15 features
```

If this happens — **don't panic.** Just:
1. Go back to Colab
2. Run `print(list(X_train.columns))`
3. Paste that output here in this chat
4. I'll rewrite your app.py in 2 minutes to match exactly

---

## Summary of What You Did
```
Colab  → Saved and downloaded .pkl files        ✅
GitHub → Uploaded .pkl files                    ✅
GitHub → Created app.py and requirements.txt    ✅
Streamlit Cloud → Deployed live app             ✅
  
