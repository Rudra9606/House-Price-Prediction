# House Price Prediction App

A Streamlit-based web application for predicting house prices in Bengaluru using machine learning.

## Local Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run House_Price_prediction/app.py
   ```

5. Open your browser to `http://localhost:8501`

## Deployment on Railway

### Prerequisites
- GitHub account with the repository pushed
- Railway account (https://railway.app)

### Steps

1. **Push your code to GitHub**
   - Create a new GitHub repository
   - Push this project to GitHub

2. **Connect to Railway**
   - Go to https://railway.app
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Authorize GitHub and select this repository

3. **Configure Environment (Optional)**
   - Railway will automatically detect the Procfile
   - The app will use port `$PORT` from Railway's environment

4. **Deploy**
   - Railway will automatically build and deploy your app
   - Your app will be live at the provided Railway URL

### Important Notes
- The app loads model files from the `House_Price_prediction/` directory
- Ensure `House_Price_predict_model.pkl` and `Bengaluru_House_Data.csv` are included in the repository
- The `.gitignore` file prevents unnecessary files from being deployed

## Files

- `House_Price_prediction/app.py` - Main Streamlit application
- `House_Price_prediction/House_Price_predict_model.pkl` - Pre-trained ML model
- `House_Price_prediction/Bengaluru_House_Data.csv` - Training dataset
- `requirements.txt` - Python dependencies
- `Procfile` - Railway deployment configuration
- `.streamlit/config.toml` - Streamlit configuration

## Features

- Select location from available categories
- Input house specifications (area, bedrooms, bathrooms, balconies)
- Get instant price prediction
- Responsive web interface
