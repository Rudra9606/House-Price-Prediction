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
csv_path = os.path.join(script_dir, 'Bengaluru_House_Data.csv')
data = pd.read_csv(csv_path)

# Get location categories from model's transformer
loc_categories = None
try:
    # Try different ways to extract categories based on model structure
    ct = model.named_steps['columntransformer']
    
    # Get the first transformer (should be for location)
    transformer = ct.transformers_[0][1]
    
    # Try to get OneHotEncoder
    if hasattr(transformer, 'named_steps'):
        ohe = transformer.named_steps.get('onehotencoder')
        if ohe and hasattr(ohe, 'categories_'):
            loc_categories = ohe.categories_[0]
    elif hasattr(transformer, 'categories_'):
        loc_categories = transformer.categories_[0]
    
    if loc_categories is not None:
        loc_categories = sorted(loc_categories.tolist())
except Exception as e:
    pass

# If we couldn't get from model, use data but filtered
if loc_categories is None:
    try:
        loc_categories = sorted(data['location'].dropna().unique().tolist())
    except:
        loc_categories = ['Unknown']
        st.error("Could not load locations. Please check the data file.")

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
        output = model.predict(input_data)
        price = float(output[0])
        
        # Display result
        out_str = f'Price of the house is: ₹{price:,.2f}'
        st.success(out_str)
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please ensure the location is valid and all inputs are numeric.")
