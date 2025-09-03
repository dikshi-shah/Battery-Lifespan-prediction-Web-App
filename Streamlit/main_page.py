import streamlit as st 
import numpy as np 
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf 
import keras
from sklearn.decomposition import PCA
import pickle


# --- MAIN PAGE ---
# st.theme.
st.set_page_config(
    page_title="Battery Lifespan Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🔋 Battery Lifespan Dashboard")

# Load data
data_load_state = st.text('Loading data...')
data = pd.read_csv('Battery_RUL.csv')
data_load_state.text("Data Loading Done!")
st.dataframe(data)



# --- SIDEBAR ---
st.sidebar.title("Settings")

# Create labeled ranges in steps of 10
min_val = int(data['Cycle_Index'].min())
max_val = int(data['Cycle_Index'].max())
range_labels = [f"{0}-{i+9}" for i in range(min_val, max_val,10)]

# Dropdown to select range
filter_value = st.sidebar.selectbox('Select Cycle Index Range',range_labels)
# Convert selected label to numeric start and end
start, end = map(int, filter_value.split('-'))
filtered_data = data[(data['Cycle_Index'] <= end)]

# Making cycle groups
if ( filtered_data['Cycle_Index'].max() <= 50):
    filtered_data['Cycle_Group'] = (filtered_data['Cycle_Index'] // 1)*1
elif(50 <= filtered_data['Cycle_Index'].max() <= 100):
    filtered_data['Cycle_Group'] = (filtered_data['Cycle_Index'] //5)*5
elif(100 <=filtered_data['Cycle_Index'].max() <= 500): 
    filtered_data['Cycle_Group'] = (filtered_data['Cycle_Index'] // 10)*10
elif(500<= filtered_data['Cycle_Index'].max()):
    filtered_data['Cycle_Group'] = (filtered_data['Cycle_Index'] // 20)*20
else:
    print('selection out of range :(')

grouped_data = filtered_data.groupby('Cycle_Group', as_index=False)[['Charging time (s)', 'Discharge Time (s)']].mean()

# ---Chart---

fig = px.line(
    grouped_data,
    x='Cycle_Group',
    y=['Charging time (s)', 'Discharge Time (s)'],
    title='Charging and Discharge Time Analysis',
    labels={
        'Cycle_Group': 'Cycle',
        'Charging time (s)': 'Charging Time (s)',
        'Discharge Time (s)': 'Discharge Time (s)'
    }
)

st.plotly_chart(fig, use_container_width=True)


# # --- MAIN PAGE ---##
st.title("Battery Lifespan Dashboard")

# # --- METRICS --- ##
col1, col2, col3 = st.columns(3)
col2.metric("Recent Decrement 3.6-3.4V (s)", f"{filtered_data['Decrement 3.6-3.4V (s)'].iloc[-1]:.2f} s")
col3.metric("Recent Max. Voltage Dischar. (V)", f"{filtered_data['Max. Voltage Dischar. (V)'].iloc[-1]:.2f} V")
col1.metric("Recent Discharge time(s)", f"{filtered_data['Discharge Time (s)'].iloc[-1]:.2f} s")

st.sidebar.header('Choose Analysis Feature')
option = st.sidebar.radio(
    'Choose Voltage Range:',
    ['Max. Voltage Dischar. (V)', 'Decrement 3.6-3.4V (s)','Min. Voltage Charg. (V)']
)
# --- CHARTS --- #
    #-- Voltage and discrement changes--#

st.subheader("Voltage Drop and Discharge Trends")

if option == "Max. Voltage Dischar. (V)":
    fig = px.scatter(filtered_data, x="Cycle_Group", y=["Max. Voltage Dischar. (V)"],
                  labels={'Cycle_Group': 'Cycle Group', 'variable': 'Metric'},
                  title="Battery Performance Over Time")
elif option == "Decrement 3.6-3.4V (s)":   
    fig = px.scatter(filtered_data, x="Cycle_Group", y=["Decrement 3.6-3.4V (s)"],
                    #  color="variable",
                     labels={'Cycle_Group': 'Cycle Group', 'variable': 'Metric'},
                     title="Battery Performance Over Time")
else:
    fig = px.scatter(filtered_data, x="Cycle_Group", y=["Decrement 3.6-3.4V (s)"],
                    #  color="variable",
                     labels={'Cycle_Group': 'Cycle Group', 'variable': 'Metric'},
                     title="Battery Performance Over Time")
st.plotly_chart(fig, use_container_width=True)

    # -- "Time Spent at 4.15V and in Constant Current Mode" -- #
st.subheader("Time Spent at 4.15V and in Constant Current Mode")

chart_type = st.sidebar.selectbox("Select a Time-Based Feature", ["Time at 4.15V (s)", "Time constant current (s)"])

if chart_type == "Time at 4.15V (s)":
    fig = px.bar(filtered_data, x="Cycle_Group", y=["Time at 4.15V (s)"],
                  labels={'Cycle_Group': 'Cycle Group', 'variable': 'Metric'},
                  title="Time At 4.15V Trend")
else:   
    fig = px.bar(filtered_data, x="Cycle_Group", y=["Time constant current (s)"],
                     color="variable",
                     labels={'Cycle_Group': 'Cycle Group', 'variable': 'Metric'},
                     title="Time Constant Current Analysis")

st.plotly_chart(fig, use_container_width=True)

# Predict button

if st.sidebar.button('Make prediction??'):
    st.switch_page(r'pages/predict.py') 
