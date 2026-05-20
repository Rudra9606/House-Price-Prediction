import joblib
import pandas as pd
import sys

# Load both models
try:
    model1 = joblib.load('House_Price_predict_model.pkl')
    print("Loaded House_Price_predict_model.pkl successfully")
except Exception as e:
    print(f"Error loading House_Price_predict_model.pkl: {e}")
    model1 = None

try:
    model2 = joblib.load('model.pkl')
    print("Loaded model.pkl successfully")
except Exception as e:
    print(f"Error loading model.pkl: {e}")
    model2 = None

# Load data
data = pd.read_csv('Bengaluru_House_Data.csv')
print(f"\nData shape: {data.shape}")
print(f"Data columns: {list(data.columns)}")

# Try to make a prediction with first row
print("\n--- Testing with first data row ---")
test_input = data.iloc[:1].copy()
print(f"Test input shape: {test_input.shape}")
print(f"Test input columns: {list(test_input.columns)}")

if model1:
    try:
        pred = model1.predict(test_input)
        print(f"Model 1 prediction successful: {pred[0]}")
    except Exception as e:
        print(f"Model 1 prediction failed: {type(e).__name__}: {e}")

if model2:
    try:
        pred = model2.predict(test_input)
        print(f"Model 2 prediction successful: {pred[0]}")
    except Exception as e:
        print(f"Model 2 prediction failed: {type(e).__name__}: {e}")
