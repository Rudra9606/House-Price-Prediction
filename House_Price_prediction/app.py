import pandas as pd
import streamlit as st
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title('🏠 House Price Prediction')
st.markdown("---")

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read model categories from file
categories_file = os.path.join(script_dir, 'model_categories.txt')
try:
    with open(categories_file, 'r') as f:
        loc_categories = [line.strip() for line in f.readlines() if line.strip()]
except:
    # Fallback: try to load from CSV
    csv_path = os.path.join(script_dir, 'Bengaluru_House_Data.csv')
    data = pd.read_csv(csv_path)
    loc_categories = sorted(data['location'].dropna().unique().tolist())

# Load training data for average price calculation
csv_path = os.path.join(script_dir, 'Bengaluru_House_Data.csv')
try:
    data = pd.read_csv(csv_path)
    # Create a simple model based on average prices
    avg_prices = data.groupby('location')['price'].mean().to_dict()
except:
    avg_prices = {}

st.header('Enter the details of the house')

col1, col2 = st.columns(2)

with col1:
    loc = st.selectbox('📍 Location', loc_categories)
    sqft = st.number_input('📏 Square Feet', min_value=100.0, max_value=100000.0, value=1500.0, step=100.0)

with col2:
    beds = st.number_input('🛏️ BHK (Bedrooms)', min_value=1, max_value=10, value=3, step=1)
    bath = st.number_input('🚿 Bathrooms', min_value=1, max_value=10, value=2, step=1)

balc = st.number_input('🏘️ Balcony', min_value=0, max_value=10, value=1, step=1)

st.markdown("---")

if st.button('🔮 Predict Price', use_container_width=True):
    try:
        # Simple prediction formula based on location average and features
        if loc in avg_prices:
            base_price = avg_prices[loc]
        else:
            # Use overall average if location not found
            base_price = data['price'].mean() if not data.empty else 50  # Default in lakhs
        
        # Adjust based on features (simplified model)
        # Price per sqft (in lakhs per 1000 sqft)
        price_per_sqft_factor = (sqft / 1000) * (base_price / 100)
        
        # BHK factor (more bedrooms increase price)
        bhk_factor = 1 + (beds - 1) * 0.15
        
        # Bathroom factor
        bath_factor = 1 + (bath - 2) * 0.05
        
        # Balcony factor (positive feature)
        balc_factor = 1 + (balc * 0.03)
        
        # Calculate final price
        predicted_price = base_price * (price_per_sqft_factor / (base_price / 100)) * bhk_factor * bath_factor * balc_factor
        
        # Display result
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Base Price", f"₹{base_price:.2f} L")
        with col2:
            st.metric("🏘️ Location", loc)
        with col3:
            st.metric("📐 Area", f"{sqft:.0f} sqft")
        
        st.markdown("---")
        
        st.success(f"### 🎯 Predicted Price: **₹{predicted_price:.2f} Lakhs**")
        
        st.info(f"""
        **Price Breakdown:**
        - Base location price: ₹{base_price:.2f} L
        - BHK adjustment: {bhk_factor:.2f}x
        - Bathroom adjustment: {bath_factor:.3f}x
        - Balcony factor: {balc_factor:.3f}x
        - **Final Price: ₹{predicted_price:.2f} Lakhs** (~₹{predicted_price*100000:,.0f})
        """)
        
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("Please ensure all inputs are valid numbers.")
