import streamlit as st
import pandas as pd
import joblib
import datetime
import os

# Imports needed for retraining if the model file is incompatible
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# --- Configuration ---
MODEL_FILE = 'supermarket_sales_model.pkl'
DATA_FILE = 'SuperMarket Analysis.csv'

# --- Model Training Function ---
def train_and_save_model():
    """Retrains the model using the CSV file and saves it."""
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset '{DATA_FILE}' not found. Please upload it to the application directory.")
        return None

    df = pd.read_csv(DATA_FILE)

    # Preprocessing identical to the notebook
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
        # Try loading the existing model
        model = joblib.load(MODEL_FILE)
        return model
    except (FileNotFoundError, AttributeError, ModuleNotFoundError, Exception):
        # If loading fails (version mismatch or missing file), retrain immediately
        # st.warning("Model version mismatch or file not found. Retraining model...")
        return train_and_save_model()

# Load the model (or retrain if needed)
model = load_model()

# --- Streamlit UI ---
st.title("🛒 Supermarket Sales Prediction App")
st.markdown("""
This application forecasts the **Total Sales** for a supermarket transaction.
""")

if model is None:
    st.stop() # Stop execution if model failed to load/train

# Sidebar Inputs
st.sidebar.header("Transaction Details")

# Define options (hardcoded for UI stability)
cities = ['Yangon', 'Naypyitaw', 'Mandalay']
customer_types = ['Member', 'Normal']
genders = ['Male', 'Female']
product_lines = ['Health and beauty', 'Electronic accessories', 'Home and lifestyle', 
                 'Sports and travel', 'Food and beverages', 'Fashion accessories']
payments = ['Ewallet', 'Cash', 'Credit card']

# Input Fields
branch_input = st.sidebar.text_input("Branch Name", value="A") 
city = st.sidebar.selectbox("City", options=cities)
customer_type = st.sidebar.selectbox("Customer Type", options=customer_types)
gender = st.sidebar.selectbox("Gender", options=genders)
product_line = st.sidebar.selectbox("Product Line", options=product_lines)
unit_price = st.sidebar.number_input("Unit Price ($)", min_value=1.0, max_value=500.0, value=50.0)
quantity = st.sidebar.slider("Quantity", min_value=1, max_value=50, value=1)
payment = st.sidebar.selectbox("Payment Method", options=payments)
rating = st.sidebar.slider("Customer Rating (Expected)", 4.0, 10.0, 7.0)
date = st.sidebar.date_input("Transaction Date", datetime.date(2019, 1, 1))
time = st.sidebar.time_input("Transaction Time", datetime.time(12, 00))

# Predict Button
if st.button("Predict Total Sales"):
    # Prepare input data
    input_data = pd.DataFrame({
        'Branch': [branch_input], 
        'City': [city],
        'Customer type': [customer_type],
        'Gender': [gender],
        'Product line': [product_line],
        'Unit price': [unit_price],
        'Quantity': [quantity],
        'Rating': [rating],
        'Month': [date.month],
        'Day': [date.day],
        'Hour': [time.hour],
        'Payment': [payment]
    })

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Total Sales: **${prediction:.2f}**")
        
        # Breakdown info
        subtotal = unit_price * quantity
        tax_est = prediction - subtotal
        st.info(f"Breakdown Estimate:\n- Subtotal (Price x Qty): ${subtotal:.2f}\n- Estimated Tax & Variation: ${tax_est:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit & Scikit-Learn")