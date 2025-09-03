import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="🔋 Battery Prediction", page_icon="📈")
st.title("🔍 Battery Lifespan Prediction")

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.info("Enter the following parameters for prediction:")

cycle_index = st.number_input("Cycle_Index", value=100)
discharge_time = st.number_input("Discharge Time (s)", value=500)
decrement = st.number_input("Decrement 3.6-3.4V (s)", value=30)
max_voltage = st.number_input("Max. Voltage Dischar. (V)", value=4.2)
min_voltage = st.number_input("Min. Voltage Charg. (V)", value=3.0)
time_4_15v = st.number_input("Time at 4.15V (s)", value=100)
time_constant = st.number_input("Time constant current (s)", value=80)
charging_time = st.number_input("Charging time (s)", value=300)

X = np.array([[
    cycle_index, discharge_time, decrement, max_voltage,
    min_voltage, time_4_15v, time_constant, charging_time
]])

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
scale = scaler.transform(X)


with open('PCA.pkl', 'rb') as file:
    PCA = pickle.load(file)
pca = PCA.transform(scale)

if st.button("Predict RUL"):

    prediction = model.predict(pca)
    st.success(f"🔋 Predicted Remaining Useful Life: {prediction[0]:} cycles")
