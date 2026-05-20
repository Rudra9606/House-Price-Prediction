import pandas as pd
import joblib
import streamlit as st
import pickle
import sys
import warnings
warnings.filterwarnings('ignore')

# Handle scikit-learn version compatibility
import sklearn.compose._column_transformer as ct

# Define _RemainderColsList if it doesn't exist (for newer scikit-learn versions)
if not hasattr(ct, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList

# Try to load model with compatibility for different scikit-learn versions
try:
    model = joblib.load('House_Price_predict_model.pkl')
except (AttributeError, ModuleNotFoundError) as e:
    # If version mismatch, try alternative model file
    try:
        model = joblib.load('model.pkl')
    except:
        st.error(f"Unable to load model: {str(e)}")
        st.stop()
st.title('House Price Prediction')

st.header('Enter the details of the house')
data = pd.read_csv('Bengaluru_House_Data.csv')

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
