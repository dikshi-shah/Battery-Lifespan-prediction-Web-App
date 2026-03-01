from pathlib import Path
import streamlit as st
import numpy as np
import pickle

STREAMLIT_DIR = Path(__file__).resolve().parent.parent 
MODEL_PATH = STREAMLIT_DIR / "Model.pkl"
SCALER_PATH = STREAMLIT_DIR / "scaler.pkl" 
PCA_PATH = STREAMLIT_DIR / "PCA.pkl"

st.set_page_config(page_title="🔋 Battery Prediction", page_icon="📈")
st.title("🔍 Battery Lifespan Prediction")

# Load all artifacts once 
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH ,"rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(PCA_PATH, "rb") as f:
        pca_transformer = pickle.load(f)
    return model, scaler, pca_transformer
try: 
    model, scaler, pca_transformer = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Model artifacts not found. Train the model using `battery_life_DL.ipynb` "
        "and place `Model.pkl`, `scaler.pkl`, and `PCA.pkl` in the `Streamlit/` folder."
    )
    st.stop()


st.info("Enter the following parameters for prediction:")

cycle_index = st.number_input("Cycle_Index", value=100)
discharge_time = st.number_input("Discharge Time (s)", value=500)
decrement = st.number_input("Decrement 3.6-3.4V (s)", value=30)
max_voltage = st.number_input("Max. Voltage Dischar. (V)", value=4.2)
min_voltage = st.number_input("Min. Voltage Charg. (V)", value=3.0)
time_4_15v = st.number_input("Time at 4.15V (s)", value=100)
time_constant = st.number_input("Time constant current (s)", value=80)
charging_time = st.number_input("Charging time (s)", value=300)

if st.button("Predict RUL"):
    X = np.array([[
        cycle_index, discharge_time, decrement, max_voltage,
        min_voltage, time_4_15v, time_constant, charging_time
    ]])
    X_scaled = scaler.transform(X)
    X_pca = pca_transformer.transform(X_scaled)
    prediction = model.predict(X_pca)
    st.success(f"🔋 Predicted Remaining Useful Life: {prediction[0]:.0f} cycles")



