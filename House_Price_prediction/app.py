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

try:
    loc_categories = model.named_steps['columntransformer'].transformers_[0][1].named_steps['onehotencoder'].categories_[0]
except Exception:
    loc_categories = data['location'].unique().tolist()

loc = st.selectbox('Location', loc_categories)
sqft = st.number_input('Square Feet', min_value=0, step=1)
beds = st.number_input('BHK', min_value=1, step=1)
bath = st.number_input('Bathroom', min_value=1, step=1)
balc = st.number_input('Balcony', min_value=0, step=1)

input = pd.DataFrame([[loc, sqft, beds, bath, balc]], columns=['location', 'total_sqft', 'bedrooms', 'bath', 'balcony'])

if st.button('Predict Price'):
    if loc not in loc_categories:
        st.error(f"Location '{loc}' not recognized by the model. Please select a valid location.")
    else:
        output = model.predict(input)
        out_str = 'Price of the house is: ' + str(output[0]*100000)
        st.success(out_str)
