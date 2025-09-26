import streamlit as st
import pandas as pd
import joblib

# Load trained model and states
model = joblib.load("sales_predictor.pkl")
states = joblib.load("states.pkl")

st.title("📊 State Revenue Predictor")

# Dropdown for state
state = st.selectbox("Select a State", states)

# Number input for year
year = st.number_input("Enter Year", min_value=2010, max_value=2100, step=1)

# Predict button
if st.button("Predict Revenue"):
    new_data = pd.DataFrame({
        "State": [state],
        "Year": [year]
    })
    prediction = model.predict(new_data)[0]
    st.success(f"Predicted Revenue for {state} in {year}: ${prediction:,.2f}")
