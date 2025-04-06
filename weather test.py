import os
import fastf1
import pandas as pd
import streamlit as st
import requests
from datetime import datetime, timedelta

# Use your existing OpenWeather API key (from context)
API_KEY = '9902ad598ba458f05379b0deb1f086b7'  # Replace this with your actual API key or Streamlit secret

# Set up FastF1 cache
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

st.title("F1 Race Weather Forecast App 🌤️")

# Step 1: Build list of race locations using FastF1
@st.cache_data
def get_race_locations():
    locations = set()
    for year in range(2018, 2025):
        for rnd in range(1, 25):
            try:
                session = fastf1.get_session(year, rnd, 'R')
                session.load()
                event_location = session.event['EventName']
                locations.add(event_location)
            except Exception:
                continue
    return sorted(list(locations))

race_locations = get_race_locations()

# User selects race location from dropdown
location = st.selectbox("Select the race location:", race_locations)

# User selects race date
selected_date = st.date_input("Select the race date:", value=datetime.today() + timedelta(days=1))

# Convert selected date to timestamp for filtering
selected_timestamp = datetime.combine(selected_date, datetime.min.time()).timestamp()

if st.button("Get Weather Forecast"):
    try:
        # Step 2: Get location coordinates
        geocode_url = f'http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={API_KEY}'
        geocode_response = requests.get(geocode_url)
        geocode_data = geocode_response.json()

        if not geocode_data:
            st.error("❌ Location not found in OpenWeather API. Please check spelling or try a nearby city.")
        else:
            lat = geocode_data[0]['lat']
            lon = geocode_data[0]['lon']

            # Step 3: Get weather forecast
            forecast_url = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
            forecast_response = requests.get(forecast_url)
            forecast_data = forecast_response.json()

            # Step 4: Filter forecast data for selected date
            forecast_list = forecast_data.get('list', [])

            filtered_forecast = []
            for entry in forecast_list:
                entry_time = datetime.fromtimestamp(entry['dt'])
                if entry_time.date() == selected_date:
                    filtered_forecast.append({
                        'Time': entry_time.strftime("%H:%M"),
                        'Temperature (°C)': entry['main']['temp'],
                        'Weather': entry['weather'][0]['description'].title(),
                        'Wind Speed (m/s)': entry['wind']['speed'],
                        'Humidity (%)': entry['main']['humidity'],
                        'Rain (mm)': entry.get('rain', {}).get('3h', 0)
                    })

            if filtered_forecast:
                st.success(f"✅ Weather forecast for {location} on {selected_date}:")
                st.table(filtered_forecast)
            else:
                st.warning("⚠️ No forecast data available for the selected date (OpenWeather only supports ~5 days ahead).")

    except Exception as e:
        st.error(f"❌ Error fetching weather data: {e}")
