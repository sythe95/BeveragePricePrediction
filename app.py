import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from PIL import Image

# Load the saved model and preprocessing objects
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("product_encoding.pkl", "rb") as f:
    product_encoding = pickle.load(f)
with open("region_encoding.pkl", "rb") as f:
    region_encoding = pickle.load(f)

# Streamlit App Title
st.set_page_config(page_title="Beverage Sales Price Prediction", page_icon="🍹", layout="centered")

# Display logo
logo = Image.open("logo.jpeg")
st.image(logo, use_column_width=True)

st.title("Beverage Sales Price Prediction")

# Collect user input
st.subheader("Enter the Features")
customer_type = st.selectbox("Customer Type", ["B2B", "B2C"])
unit_price = st.number_input("Unit Price", min_value=0.0, step=0.1)
quantity = st.number_input("Quantity", min_value=1, step=1)
discount = st.number_input("Discount", min_value=0.0, step=0.1)
product = st.selectbox("Product Name", list(product_encoding.keys()))
region = st.selectbox("Region Name", list(region_encoding.keys()))
year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
month = st.slider("Month", min_value=1, max_value=12, step=1)
day = st.slider("Day", min_value=1, max_value=31, step=1)
day_of_week = st.slider("Day of Week (0: Monday - 6: Sunday)", min_value=0, max_value=6, step=1)
quarter = st.slider("Quarter", min_value=1, max_value=4, step=1)

# Encoding
customer_type_encoded = 1 if customer_type == "B2B" else 0
product_encoded = product_encoding.get(product, np.mean(list(product_encoding.values())))
region_encoded = region_encoding.get(region, np.mean(list(region_encoding.values())))

# Create a dataframe for prediction
user_data = pd.DataFrame([[customer_type_encoded, unit_price, quantity, discount, product_encoded, region_encoded, year, month, day, day_of_week, quarter]],
                         columns=["Customer_Type", "Unit_Price", "Quantity", "Discount", "Product", "Region", "Year", "Month", "Day", "Day_of_Week", "Quarter"])

# Apply scaling
user_data_scaled = scaler.transform(user_data)

# Predict button
if st.button("Predict Total Price"):
    prediction = model.predict(user_data_scaled)
    predicted_price = np.expm1(prediction[0])  # Reverse log transformation
    st.success(f"Predicted Total Price: ${predicted_price:.2f}")
