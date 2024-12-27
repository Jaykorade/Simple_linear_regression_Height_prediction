import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the pre-trained model
with open('regressor.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title('Height Prediction App')

# Initialize StandardScaler
scaler = StandardScaler()

# Input weight
weight = st.number_input('Enter weight (kg)', min_value=0.0, step=0.1)

# Convert weight to a 2D array
weight_array = np.array([[weight]])  # 2D array

# Fit the scaler with a dummy example (normally you fit it during model training)
scaler.fit(weight_array)

# Predict height
if st.button('Predict Height'):
    # Scale the weight
    scaled_weight = scaler.transform(weight_array)
    
    # Predict height
    height = model.predict(scaled_weight)
    
    st.write(f'Predicted Height: {float(height[0]):.2f}')
