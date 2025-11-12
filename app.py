import streamlit as st
import numpy as np
import pickle



# Load your trained Random Forest model
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App Title & Info
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

st.title("🏠 California House Price Prediction App")
st.markdown("""
This app predicts **house prices** using a trained Random Forest Regression model.  
Please enter the required housing details below ⬇️
""")

# --- INPUT FIELDS ---
longitude = st.number_input("Longitude", format="%.6f")
latitude = st.number_input("Latitude", format="%.6f")
housing_median_age = st.number_input("Housing Median Age", min_value=1.0)
total_rooms = st.number_input("Total Rooms", min_value=1.0)
total_bedrooms = st.number_input("Total Bedrooms", min_value=1.0)
population = st.number_input("Population", min_value=1.0)
households = st.number_input("Households", min_value=1.0)
median_income = st.number_input("Median Income", min_value=0.0)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ("<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN")
)

# --- ENCODE CATEGORICAL FEATURE ---
ocean_dict = {
    "<1H OCEAN": [1, 0, 0, 0, 0],
    "INLAND": [0, 1, 0, 0, 0],
    "ISLAND": [0, 0, 1, 0, 0],
    "NEAR BAY": [0, 0, 0, 1, 0],
    "NEAR OCEAN": [0, 0, 0, 0, 1],
}
ocean_features = ocean_dict[ocean_proximity]

# --- APPLY SAME PREPROCESSING AS NOTEBOOK ---
total_rooms_log = np.log(total_rooms + 1)
total_bedrooms_log = np.log(total_bedrooms + 1)
population_log = np.log(population + 1)
households_log = np.log(households + 1)

bedroom_ratio = total_bedrooms_log / total_rooms_log
household_rooms = total_rooms_log / households_log

# --- FINAL INPUT VECTOR ---
features = np.array([[
    longitude, latitude, housing_median_age,
    total_rooms_log, total_bedrooms_log, population_log, households_log,
    median_income, bedroom_ratio, household_rooms,
    *ocean_features
]])

# --- PREDICTION BUTTON ---
if st.button("💰 Predict House Price"):
    prediction = model.predict(features)
    st.success(f"🏡 Estimated Median House Price: **${prediction[0]:,.2f}**")

    st.balloons()
