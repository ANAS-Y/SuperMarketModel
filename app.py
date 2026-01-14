import streamlit as st
import pandas as pd
import joblib
import datetime
import os
import numpy as np

# Imports needed for retraining
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- Configuration ---
MODEL_FILE = 'supermarket_sales_model.pkl'
DATA_FILE = 'Nigerian_Supermarket_Analysis.csv'

# --- Model Training Function ---
def train_and_save_model():
    """Retrains the model using the Nigerian dataset and saves it."""
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset '{DATA_FILE}' not found. Please ensure 'process_dataset.py' has been run.")
        return None

    df = pd.read_csv(DATA_FILE)

    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Hour'] = pd.to_datetime(df['Time']).dt.hour

    # Features and Target
    X = df.drop(['Invoice ID', 'Tax 5%', 'Sales', 'cogs', 'gross margin percentage', 
                 'gross income', 'Date', 'Time'], axis=1)
    y = df['Sales']

    categorical_cols = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']
    numerical_cols = ['Unit price', 'Quantity', 'Rating', 'Month', 'Day', 'Hour']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ])

    # Pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', RandomForestRegressor(n_estimators=100, random_state=42))])
    
    # Fit
    model.fit(X, y)
    
    # Save
    joblib.dump(model, MODEL_FILE)
    return model

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except (FileNotFoundError, AttributeError, ModuleNotFoundError, Exception):
        return train_and_save_model()

model = load_model()

# --- Streamlit UI ---
st.title("🛒 Supermarket Sales Prediction")
st.markdown("""
This app predicts the **Total Sales** for a transaction. 
Enter the key product details below.
""")

if model is None:
    st.stop()

# --- Inputs ---
st.sidebar.header("Key Transaction Details")

# 1. Location Context (City determines Branch)
city_branch_map = {
    'Lagos': 'Ikeja',
    'Abuja': 'Maitama',
    'Port Harcourt': 'GRA'
}
cities = list(city_branch_map.keys())
selected_city = st.sidebar.selectbox("City", options=cities)
inferred_branch = city_branch_map[selected_city] # Auto-select branch based on city

# 2. Product Context
product_lines = ['Health and beauty', 'Electronic accessories', 'Home and lifestyle', 
                 'Sports and travel', 'Food and beverages', 'Fashion accessories']
selected_product_line = st.sidebar.selectbox("Product Line", options=product_lines)

# 3. The Major Predictors (Unit Price & Quantity)
st.sidebar.markdown("---")
st.sidebar.subheader("Sales Drivers")
unit_price = st.sidebar.number_input("Unit Price (₦)", min_value=1.0, max_value=1000.0, value=50.0)
quantity = st.sidebar.slider("Quantity", min_value=1, max_value=50, value=1)

# 4. Date & Time Inputs (Restored for Forecasting)
st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Date")
transaction_date = st.sidebar.date_input("Date", datetime.date.today())
# We include time because 'Hour' is a feature in the model
transaction_time = st.sidebar.time_input("Time (Approx)", datetime.datetime.now().time())

# --- Hidden Defaults for "Less Important" Features ---
# These are required by the model but have low impact on the specific calculation
default_customer = 'Normal'
default_gender = 'Female'
default_payment = 'Cash'
default_rating = 7.0

# Predict Button
if st.button("Predict Total Sales"):
    # Prepare input data with User Inputs + Defaults
    input_data = pd.DataFrame({
        'Branch': [inferred_branch], 
        'City': [selected_city],
        'Customer type': [default_customer],
        'Gender': [default_gender],
        'Product line': [selected_product_line],
        'Unit price': [unit_price],
        'Quantity': [quantity],
        'Rating': [default_rating],
        'Month': [transaction_date.month],
        'Day': [transaction_date.day],
        'Hour': [transaction_time.hour],
        'Payment': [default_payment]
    })

    try:
        prediction = model.predict(input_data)[0]
        
        # Display Result
        st.success(f"💰 Predicted Total Sales: **₦{prediction:.2f}**")
        
        # Contextual info
        st.caption(f"📍 Location: {inferred_branch}, {selected_city}")
        
        # Breakdown
        subtotal = unit_price * quantity
        tax_est = prediction - subtotal
        st.info(f"**Breakdown:**\n- Subtotal: ₦{subtotal:.2f}\n- VAT (5%): ₦{tax_est:.2f}")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.markdown("---")
st.markdown("Developed by Anas Yunusa Adamu|3MTT Fellow for Cohort 1 DeepTech ")