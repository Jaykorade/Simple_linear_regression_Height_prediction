import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Load the pre-trained model
with open('regressor.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title('Height Prediction App')

# Input weight
weight = st.number_input('Enter weight (kg)', min_value=0.0, step=0.1)
#weight = scaler.fit_transform(weight)
# Predict height
if st.button('Predict Height'):
    height = model.predict([[weight]])
    st.write(f'Predicted Height: {height}')
