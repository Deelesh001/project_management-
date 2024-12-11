import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open("trained_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'trained_model.pkl' is available.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Initialize a session state to save scores
if "scores" not in st.session_state:
    st.session_state["scores"] = []

# App title
st.title("Vehicle Sustainability Predictor")

# Sidebar for user input
st.sidebar.header("Input Vehicle Attributes")

# Collect user input
def user_input():
    try:
        fuel_type = st.sidebar.selectbox("Fuel Type", options=["Gasoline", "Diesel", "Electric", "Hybrid"])

        # Common inputs
        inputs = {
            "Mh": st.sidebar.text_input("Manufacturer (e.g., Toyota, Tesla, BMW)", value="Toyota"),
            "Cn": st.sidebar.text_input("Commercial Name (e.g., Model S, Corolla)", value="Corolla"),
            "Fuel Type": fuel_type,
            "m (kg)": st.sidebar.number_input("Vehicle Mass (kg)", min_value=500.0, max_value=5000.0, step=100.0, value=1500.0),
            "ep (KW)": st.sidebar.number_input("Engine Power (KW)", min_value=0.0, max_value=500.0, step=10.0, value=100.0),
            "Recycling Potential": st.sidebar.number_input("Recycling Potential (%)", min_value=0.0, max_value=100.0, step=5.0, value=50.0),
            "Enedc (g/km)": 0,
            "Ewltp (g/km)": 0,
            "ec (cm3)": 0,
            "Fuel consumption ": 0,
            "Electric range (km)": 0,
            "Carbon Intensity (gCO2/kWh)": 0,
            "Lifecycle Emissions (gCO2/km)": 0
        }

        # Conditional inputs based on fuel type
        if fuel_type in ["Gasoline", "Diesel"]:
            inputs.update({
                "Enedc (g/km)": st.sidebar.number_input("Emissions NEDC (g/km)", min_value=0.0, max_value=400.0, step=10.0, value=100.0),
                "Ewltp (g/km)": st.sidebar.number_input("Emissions WLTP (g/km)", min_value=0.0, max_value=400.0, step=10.0, value=120.0),
                "ec (cm3)": st.sidebar.number_input("Engine Capacity (cm3)", min_value=0.0, max_value=6000.0, step=100.0, value=2000.0),
                "Fuel consumption ": st.sidebar.number_input("Fuel Consumption (L/100km)", min_value=0.0, max_value=50.0, step=1.0, value=6.0)
            })
        elif fuel_type == "Electric":
            inputs.update({
                "Electric range (km)": st.sidebar.number_input("Electric Range (km)", min_value=0.0, max_value=500.0, step=10.0, value=300.0),
                "Carbon Intensity (gCO2/kWh)": st.sidebar.number_input("Carbon Intensity (gCO2/kWh)", min_value=0.0, max_value=1000.0, step=10.0, value=50.0)
            })
        elif fuel_type == "Hybrid":
            inputs.update({
                "Ewltp (g/km)": st.sidebar.number_input("Emissions WLTP (g/km)", min_value=0.0, max_value=400.0, step=10.0, value=80.0),
                "Electric range (km)": st.sidebar.number_input("Electric Range (km)", min_value=0.0, max_value=500.0, step=10.0, value=50.0),
                "Fuel consumption ": st.sidebar.number_input("Fuel Consumption (L/100km)", min_value=0.0, max_value=50.0, step=1.0, value=5.0)
            })

        return pd.DataFrame([inputs])
    except Exception as e:
        st.error(f"An error occurred while collecting user input: {e}")
        return pd.DataFrame()

# Get user input data
input_data = user_input()

# Display input data
if not input_data.empty:
    st.subheader("Input Data")
    st.write(input_data)

    # Predict sustainability score
    if st.button("Predict Sustainability Score"):
        try:
            prediction = model.predict(input_data)  # Ensure all columns are present with default values
            score = prediction[0]
            st.session_state["scores"].append({
                "Manufacturer": input_data.loc[0, "Mh"],
                "Model": input_data.loc[0, "Cn"],
                "Sustainability Score": score
            })
            st.subheader("Predicted Sustainability Score")
            st.write(f"{score:.2f}")
        except ValueError as ve:
            st.error(f"Value Error: {ve}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Display saved scores
st.subheader("Saved Sustainability Scores")
if st.session_state["scores"]:
    try:
        scores_df = pd.DataFrame(st.session_state["scores"])
        st.write(scores_df)

        # Compare scores
        if len(st.session_state["scores"]) > 1:
            st.subheader("Comparison")
            st.line_chart(scores_df.set_index("Model")["Sustainability Score"])
    except Exception as e:
        st.error(f"An error occurred while displaying scores: {e}")
else:
    st.write("No scores saved yet.")

# Footer
st.write("\n\n")
st.info("Model trained using Gradient Boosting Regressor with features tailored for vehicle sustainability prediction.")
