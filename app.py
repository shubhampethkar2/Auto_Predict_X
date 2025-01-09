import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import tkinter as tk
from tkinter import messagebox

# Load the model and label encoders
model = joblib.load('xgb_model.joblib')
le_dict = joblib.load('label_encoders.joblib')

# Define the categorical features
categorical_features = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']

def predict_price(data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data])

    # Encode categorical features
    for cat_feature in categorical_features:
        le = le_dict[cat_feature]
        input_df[cat_feature] = le.transform(input_df[cat_feature].astype(str))
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    return float(prediction)

def on_predict():
    try:
        # Gather input data from the form
        sample_data = {
            'brand': brand_entry.get(),
            'model': model_entry.get(),
            'vehicle_age': int(vehicle_age_entry.get()),
            'km_driven': int(km_driven_entry.get()),
            'seller_type': seller_type_entry.get(),
            'fuel_type': fuel_type_entry.get(),
            'transmission_type': transmission_type_entry.get(),
            'mileage': float(mileage_entry.get()),
            'engine': float(engine_entry.get()),
            'max_power': float(max_power_entry.get()),
            'seats': int(seats_entry.get())
        }

        # Predict price
        predicted_price = predict_price(sample_data)
        messagebox.showinfo("Predicted Price", f"The predicted price is: {predicted_price:.2f}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the main window
root = tk.Tk()
root.title("Car Price Prediction")

# Create input fields
tk.Label(root, text="Brand").grid(row=0, column=0)
brand_entry = tk.Entry(root)
brand_entry.grid(row=0, column=1)

tk.Label(root, text="Model").grid(row=1, column=0)
model_entry = tk.Entry(root)
model_entry.grid(row=1, column=1)

tk.Label(root, text="Vehicle Age (years)").grid(row=2, column=0)
vehicle_age_entry = tk.Entry(root)
vehicle_age_entry.grid(row=2, column=1)

tk.Label(root, text="KM Driven").grid(row=3, column=0)
km_driven_entry = tk.Entry(root)
km_driven_entry.grid(row=3, column=1)

tk.Label(root, text="Seller Type").grid(row=4, column=0)
seller_type_entry = tk.Entry(root)
seller_type_entry.grid(row=4, column=1)

tk.Label(root, text="Fuel Type").grid(row=5, column=0)
fuel_type_entry = tk.Entry(root)
fuel_type_entry.grid(row=5, column=1)

tk.Label(root, text="Transmission Type").grid(row=6, column=0)
transmission_type_entry = tk.Entry(root)
transmission_type_entry.grid(row=6, column=1)

tk.Label(root, text="Mileage (km/l)").grid(row=7, column=0)
mileage_entry = tk.Entry(root)
mileage_entry.grid(row=7, column=1)

tk.Label(root, text="Engine (cc)").grid(row=8, column=0)
engine_entry = tk.Entry(root)
engine_entry.grid(row=8, column=1)

tk.Label(root, text="Max Power (bhp)").grid(row=9, column=0)
max_power_entry = tk.Entry(root)
max_power_entry.grid(row=9, column=1)

tk.Label(root, text="Seats").grid(row=10, column=0)
seats_entry = tk.Entry(root)
seats_entry.grid(row=10, column=1)

# Create a predict button
predict_button = tk.Button(root, text="Predict Price", command=on_predict)
predict_button.grid(row=11, columnspan=2)

# Run the application
root.mainloop()
