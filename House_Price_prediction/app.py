import pandas as pd
import joblib
import streamlit as st
import os
import warnings
import sys
warnings.filterwarnings('ignore')

# Fix for scikit-learn version compatibility
try:
    from sklearn.compose._column_transformer import _RemainderColsList
except ImportError:
    # Define the missing class for newer scikit-learn versions
    import sklearn.compose._column_transformer as ct
    if not hasattr(ct, '_RemainderColsList'):
        class _RemainderColsList(list):
            pass
        ct._RemainderColsList = _RemainderColsList

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model with error handling
try:
    model_path = os.path.join(script_dir, 'House_Price_predict_model.pkl')
    model = joblib.load(model_path)
except Exception as e:
    try:
        # Fallback to alternative model file
        model_path = os.path.join(script_dir, 'model.pkl')
        model = joblib.load(model_path)
    except Exception:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

st.title('House Price Prediction')

st.header('Enter the details of the house')

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

loc = st.selectbox('Location', loc_categories)
sqft = st.number_input('Square Feet', min_value=0.0, step=1.0)
beds = st.number_input('BHK', min_value=1, step=1)
bath = st.number_input('Bathroom', min_value=1, step=1)
balc = st.number_input('Balcony', min_value=0, step=1)

if st.button('Predict Price'):
    try:
        # Create input dataframe with correct data types
        input_data = pd.DataFrame({
            'location': [loc],
            'total_sqft': [float(sqft)],
            'bedrooms': [int(beds)],
            'bath': [int(bath)],
            'balcony': [int(balc)]
        })
        
        # Ensure correct column order (matching training data)
        input_data = input_data[['location', 'total_sqft', 'bedrooms', 'bath', 'balcony']]
        
        # Make prediction
        try:
            output = model.predict(input_data)
            price = float(output[0])
        except ValueError as ve:
            # If feature mismatch, try model.pkl
            if "features" in str(ve):
                st.warning("Trying alternate model...")
                model_alt = joblib.load(os.path.join(script_dir, 'model.pkl'))
                output = model_alt.predict(input_data)
                price = float(output[0])
            else:
                raise
        
        # Display result
        out_str = f'Price of the house is: ₹{price:,.2f}'
        st.success(out_str)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please ensure the location is valid and all inputs are numeric.")
