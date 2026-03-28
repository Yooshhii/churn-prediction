# 📉 Customer Churn Prediction

## 🔍 Business Problem
A telecom company loses ~27% of customers every year.
The cost of acquiring a new customer is 5x higher than retaining an existing one.

This project builds a machine learning model that identifies at-risk customers
**before they leave** — enabling the business to take proactive retention action.

---

## 🚀 Live App
👉 **[Try the Churn Predictor here](https://yooshhii-churn-prediction-app-pobop9.streamlit.app/)**

Input any customer profile and get an instant churn probability with recommended actions.

---

## 📊 Dataset
- **Source:** IBM Telco Customer Churn (Kaggle)
- **Size:** 7,043 customers | 21 features
- **Target:** Whether a customer churned (Yes/No)

---

## 🔑 Key Findings from EDA
- Month-to-month contract customers churn **3x more** than annual contract holders
- Fiber optic users show the highest churn despite paying premium prices
- The **first 20 months** of tenure are the highest churn risk window
- Electronic check payment users churn more than any other payment method
- Customers without Online Security or Tech Support churn significantly more

---

## 🔬 Approach

| Step | Details |
|---|---|
| Data Cleaning | Handled nulls, fixed data types, removed irrelevant columns |
| EDA | Analysed 20+ features against churn rate |
| Feature Engineering | One-hot encoding, binary encoding, auto-computed TotalCharges |
| Modelling | Compared Logistic Regression vs XGBoost |
| Imbalance Handling | Used scale_pos_weight in XGBoost |
| Threshold Tuning | Optimised decision threshold using F1 score |
| Deployment | Streamlit app with smart conditional UI logic |

---

## 📈 Model Performance

| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | 0.822 | Weak churn probability calibration |
| XGBoost (final) | 0.83+ | Strong separation, business-tuned threshold |

**Why XGBoost won:** Logistic Regression compressed all churn probabilities below 20%.
XGBoost correctly pushed high-risk customers to 70-85% probability.

---

## 🛠 Tech Stack
![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-green)
![Streamlit](https://img.shields.io/badge/Streamlit-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-orange)

- **Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn
- **Deployment:** Streamlit Cloud
- **Version Control:** Git + GitHub

---

## 📁 Project Structure
```
churn-prediction/
│
├── app.py                  # Streamlit web application
├── churn_model.pkl         # Trained XGBoost model
├── scaler.pkl              # Fitted StandardScaler
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 💡 Business Recommendations
Based on model findings, the retention team should prioritise:
1. **Month-to-month customers in first 20 months** — offer contract upgrade incentives
2. **Fiber optic users without security add-ons** — bundle tech support at discounted rate
3. **Electronic check payers** — nudge towards auto-pay with a small discount

---

## 👤 About
Built by **Yooshhii** aka Yoshita B Shankar.
Currently working as a business analyst and looking foward to transition into ML or data science roles
Connect on LinkedIn: [https://www.linkedin.com/in/yoshita-b-shankar-686045330?utm_source=share_via&utm_content=profile&utm_medium=member_ios]
