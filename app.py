import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
wine_data = pd.read_csv('WineQT.csv')

# Selecting features and target variable
X = wine_data.drop(columns=['quality', 'Id'])  # Drop 'quality' (target) and 'Id' (identifier)
y = wine_data['quality']  # Target variable is 'quality'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Streamlit App Title
st.title("Wine Quality Prediction")

# Sidebar inputs for features with default values
st.sidebar.header("Input Features")

fixed_acidity = st.sidebar.number_input('Fixed Acidity', value=7.0)
volatile_acidity = st.sidebar.number_input('Volatile Acidity', value=0.7)
citric_acid = st.sidebar.number_input('Citric Acid', value=0.0)
residual_sugar = st.sidebar.number_input('Residual Sugar', value=2.0)
chlorides = st.sidebar.number_input('Chlorides', value=0.08)
free_sulfur_dioxide = st.sidebar.number_input('Free Sulfur Dioxide', value=30.0)
total_sulfur_dioxide = st.sidebar.number_input('Total Sulfur Dioxide', value=100.0)
density = st.sidebar.number_input('Density', value=0.996)
pH = st.sidebar.number_input('pH', value=3.3)
sulphates = st.sidebar.number_input('Sulphates', value=0.6)
alcohol = st.sidebar.number_input('Alcohol', value=10.0)

# Button to predict wine quality
if st.sidebar.button('Predict Wine Quality'):
    # Convert inputs to a numpy array for prediction
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    
    # Make prediction
    predicted_quality = rf_model.predict(input_data)[0]
    
    # Display the predicted wine quality
    st.subheader(f"Predicted Wine Quality: {predicted_quality}")
