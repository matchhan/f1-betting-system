import streamlit as st
import requests
from datetime import datetime, timedelta

# Your OpenWeather API Key
API_KEY = '9902ad598ba458f05379b0deb1f086b7'

st.title("F1 Race Weather Forecast App 🌤️")

# Clean list of F1 circuits mapped to city names for OpenWeather API
f1_locations = {
    'Albert Park, Melbourne': 'Melbourne',
    'Jeddah Corniche Circuit': 'Jeddah',
    'Bahrain International Circuit': 'Sakhir',
    'Suzuka Circuit': 'Suzuka',
    'Circuit de Monaco': 'Monaco',
    'Silverstone Circuit': 'Silverstone',
    'Circuit de Spa-Francorchamps': 'Stavelot',
    'Monza Circuit': 'Monza',
    'Marina Bay Street Circuit': 'Singapore',
    'Circuit of the Americas': 'Austin',
    'Autódromo Hermanos Rodríguez': 'Mexico City',
    'Interlagos Circuit': 'Sao Paulo',
    'Y
