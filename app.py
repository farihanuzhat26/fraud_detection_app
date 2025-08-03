import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("best_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🔍 Fraud Detection Dashboard")
st.write("Upload a CSV with the following columns: transaction_amount, Location, Age, Gender")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

top_n = st.number_input("Show top N most likely fraudulent transactions", min_value=1, step=1, value=5)

if uploaded_file:
    try:
        # Read and normalize column names
        new_data = pd.read_csv(uploaded_file)
        new_data.columns = [col.strip().lower() for col in new_data.columns]
        new_data.rename(columns={
            'transaction_amount': 'transaction_amount',
            'location': 'Location',
            'age': 'Age',
            'gender': 'Gender'
        }, inplace=True)

        # Encode categorical features using saved encoders
        for col in ['Location', 'Gender']:
            le = label_encoders[col]
            new_data[col] = le.transform(new_data[col])

        # Predict
        X_new = new_data[['transaction_amount', 'Location', 'Age', 'Gender']]
        new_data['fraud_probability'] = model.predict_proba(X_new)[:, 1]
        top_fraud = new_data.sort_values(by='fraud_probability', ascending=False).head(top_n)

        # Display
        st.write("### Top Fraudulent Transactions")
        st.dataframe(top_fraud)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
