import streamlit as st
import pandas as pd
import joblib
import datetime

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('supermarket_sales_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'supermarket_sales_model.pkl' not found. Please run the notebook first to generate the model.")
        return None

model = load_model()

# Title and Description
st.title("🛒 Supermarket Sales Prediction App")
st.markdown("""
This application forecasts the **Total Sales** for a supermarket transaction based on customer and product details.
Adjust the parameters in the sidebar to generate a prediction.
""")

# Sidebar Inputs
st.sidebar.header("Transaction Details")

# Define options based on dataset
cities = ['Yangon', 'Naypyitaw', 'Mandalay']
customer_types = ['Member', 'Normal']
genders = ['Male', 'Female']
product_lines = ['Health and beauty', 'Electronic accessories', 'Home and lifestyle', 
                 'Sports and travel', 'Food and beverages', 'Fashion accessories']
payments = ['Ewallet', 'Cash', 'Credit card']
branches = ['A', 'B', 'C'] # Generic placeholders

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
    if model is not None:
        # Prepare input data as DataFrame
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
            # Make prediction
            prediction = model.predict(input_data)[0]

            # Display Result
            st.success(f"💰 Predicted Total Sales: **${prediction:.2f}**")
            
            # Breakdown info
            subtotal = unit_price * quantity
            tax_est = prediction - subtotal
            
            st.info(f"Breakdown Estimate:\n- Subtotal (Price x Qty): ${subtotal:.2f}\n- Estimated Tax & Variation: ${tax_est:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Model not loaded.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit & Scikit-Learn")