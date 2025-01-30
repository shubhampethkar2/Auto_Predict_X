import streamlit as st
import pandas as pd
import joblib
# Load the pre-trained model and label encoders
model = joblib.load('xgb_model.joblib')
le_dict = joblib.load('label_encoders.joblib')
# Load dataset to extract dropdown options
dataset = pd.read_csv('cardekho_dataset.csv')

# Extract unique values for dropdown menus
brands = sorted(dataset['brand'].unique())
models = sorted(dataset['model'].unique())
seller_types = sorted(dataset['seller_type'].unique())
fuel_types = sorted(dataset['fuel_type'].unique())
transmissions = sorted(dataset['transmission_type'].unique())

# Streamlit UI
st.title("Predict Used Car price")

# Input fields for the car's features
selected_brand = st.selectbox("Select Brand", brands)
selected_model = st.selectbox("Select Model", models)
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
selected_seller_type = st.selectbox("Select Seller Type", seller_types)
selected_fuel_type = st.selectbox("Select Fuel Type", fuel_types)
selected_transmission = st.selectbox("Select Transmission Type", transmissions)
mileage = st.number_input("Mileage (km/l)", min_value=0.0, step=0.1)
engine = st.number_input("Engine (cc)", min_value=0.0, step=10.0)
max_power = st.number_input("Max Power (bhp)", min_value=0.0, step=0.1)
seats = st.number_input("Seats", min_value=2, max_value=10, step=1)

# Function to predict the car's price
def predict_price(data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data])

    # Encode categorical features using the label encoders
    for col in le_dict:
        input_df[col] = le_dict[col].transform(input_df[col].astype(str))
    
    # Make prediction using the model
    prediction = model.predict(input_df)[0]
    return prediction

# When the user clicks the "Predict" button
if st.button("Predict Price"):
    try:
        # Prepare the data for prediction
        sample_data = {
            'brand': selected_brand,
            'model': selected_model,
            'vehicle_age': int(vehicle_age),
            'km_driven': int(km_driven),
            'seller_type': selected_seller_type,
            'fuel_type': selected_fuel_type,
            'transmission_type': selected_transmission,
            'mileage': float(mileage),
            'engine': float(engine),
            'max_power': float(max_power),
            'seats': int(seats)
        }

        # Get the predicted price
        predicted_price = predict_price(sample_data)
        st.success(f"The predicted price is: ₹{predicted_price:,.2f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
