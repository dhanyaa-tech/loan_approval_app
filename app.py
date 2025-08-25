import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title
st.title("🏦 Loan Approval Predictor")
st.write("Enter applicant details to predict loan approval.")

# Load model and components
@st.cache_resource
def load_model():
    model = joblib.load('loan_approval_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input Form
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
        credit_score = st.number_input("Credit Score", 300, 850, 700)
        loan_amount = st.number_input("Loan Amount ($)", 1000, 500000, 10000)
        loan_duration = st.number_input("Loan Duration (months)", 6, 360, 60)
        dependents = st.number_input("Number of Dependents", 0, 10, 1)
        debt_payments = st.number_input("Monthly Debt Payments ($)", 0, 50000, 500)
        card_util = st.slider("Credit Card Utilization Rate (%)", 0.0, 100.0, 30.0) / 100

    with col2:
        open_lines = st.number_input("Number of Open Credit Lines", 0, 20, 3)
        inquiries = st.number_input("Number of Credit Inquiries", 0, 10, 1)
        dti = st.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
        bankruptcy = st.selectbox("Bankruptcy History", [0, 1], format_func=lambda x: "Yes" if x else "No")
        defaults = st.selectbox("Previous Loan Defaults", [0, 1], format_func=lambda x: "Yes" if x else "No")
        payment_history = st.slider("Payment History (0-10)", 0, 10, 8)
        job_tenure = st.number_input("Job Tenure (years)", 0, 50, 5)
        interest_rate = st.number_input("Interest Rate (%)", 1.0, 30.0, 8.5) / 100

    # Categorical inputs
    employment_status = st.selectbox("Employment Status", ["Unemployed", "Part-time", "Full-time"])
    education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    home_ownership = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"])
    loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Education", "Home", "Car", "Business"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Encode categorical variables (based on your LabelEncoder mapping)
    cat_map = {
        'Unemployed': 0, 'Part-time': 1, 'Full-time': 2,
        'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3,
        'Single': 0, 'Married': 1, 'Divorced': 2,
        'Rent': 0, 'Own': 1, 'Mortgage': 2,
        'Personal': 0, 'Education': 1, 'Home': 2, 'Car': 3, 'Business': 4
    }

    # Create input DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'AnnualIncome': income,
        'CreditScore': credit_score,
        'Experience': max(age - 22, 0),
        'LoanAmount': loan_amount,
        'LoanDuration': loan_duration,
        'NumberOfDependents': dependents,
        'MonthlyDebtPayments': debt_payments,
        'CreditCardUtilizationRate': card_util,
        'NumberOfOpenCreditLines': open_lines,
        'NumberOfCreditInquiries': inquiries,
        'DebtToIncomeRatio': dti,
        'BankruptcyHistory': bankruptcy,
        'PreviousLoanDefaults': defaults,
        'PaymentHistory': payment_history,
        'LengthOfCreditHistory': job_tenure * 12,
        'SavingsAccountBalance': income * 0.1,
        'CheckingAccountBalance': income * 0.05,
        'TotalAssets': income * 2,
        'TotalLiabilities': debt_payments * 12,
        'MonthlyIncome': income / 12,
        'UtilityBillsPaymentHistory': 0.9,
        'JobTenure': job_tenure,
        'NetWorth': income - debt_payments * 12,
        'BaseInterestRate': interest_rate,
        'InterestRate': interest_rate,
        'MonthlyLoanPayment': (loan_amount * interest_rate) / (1 - (1 + interest_rate)**(-loan_duration)),
        'TotalDebtToIncomeRatio': dti + 0.05,
        'RiskScore': credit_score / 100,
        'EmploymentStatus': cat_map[employment_status],
        'EducationLevel': cat_map[education_level],
        'MaritalStatus': cat_map[marital_status],
        'HomeOwnershipStatus': cat_map[home_ownership],
        'LoanPurpose': cat_map[loan_purpose]
    }])

    # Reorder columns to match training
    input_data = input_data[feature_names]

    # Scale input
    try:
        input_scaled = scaler.transform(input_data)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        st.stop()

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    # Show result
    if prediction == 1:
        st.success(f"✅ Loan Approved! Confidence: {probability[1]:.2f}")
    else:
        st.error(f"❌ Loan Rejected. Confidence: {probability[0]:.2f}")

    # Show top important features
    st.subheader("Key Factors")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values('Importance', ascending=False).head(5)
        st.bar_chart(imp_df.set_index('Feature'))