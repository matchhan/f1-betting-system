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
    'Yas Marina Circuit': 'Abu Dhabi',
    'Hungaroring': 'Budapest',
    'Imola Circuit': 'Imola',
    'Zandvoort Circuit': 'Zandvoort',
    'Red Bull Ring': 'Spielberg',
    'Las Vegas Street Circuit': 'Las Vegas',
    'Miami International Autodrome': 'Miami',
    'Qatar Losail Circuit': 'Lusail',
    'Baku City Circuit': 'Baku',
    'Shanghai International Circuit': 'Shanghai',
    'Circuit Gilles Villeneuve': 'Montreal',
    # Add or remove as you prefer
}

# User selects circuit
location_label = st.selectbox("Select the race circuit:", list(f1_locations.keys()))
location_city = f1_locations[location_label]

# User selects race date
selected_date = st.date_input("Select the race date:", value=datetime.today() + timedelta(days=1))

if st.button("Get Weather Forecast"):
    try:
        # Step 1: Get location coordinates from OpenWeather Geocoding API
        geocode_url = f'http://api.openweathermap.org/geo/1.0/direct?q={location_city}&limit=1&appid={API_KEY}'
        geocode_response = requests.get(geocode_url)
        geocode_data = geocode_response.json()

        if not geocode_data:
            st.error("❌ Location not found in OpenWeather. Please try a nearby city.")
        else:
            lat = geocode_data[0]['lat']
            lon = geocode_data[0]['lon']

            # Step 2: Get 5-day forecast (3-hour intervals)
            forecast_url = f'https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'
            forecast_response = requests.get(forecast_url)
            forecast_data = forecast_response.json()

            # Step 3: Filter forecast by selected date
            forecast_list = forecast_data.get('list', [])
            selected_forecast = []
            for entry in forecast_list:
                entry_time = datetime.fromtimestamp(entry['dt'])
                if entry_time.date() == selected_date:
                    selected_forecast.append({
                        'Time': entry_time.strftime("%H:%M"),
                        'Temperature (°C)': entry['main']['temp'],
                        'Weather': entry['weather'][0]['description'].title(),
                        'Wind Speed (m/s)': entry['wind']['speed'],
                        'Humidity (%)': entry['main']['humidity'],
                        'Rain (mm)': entry.get('rain', {}).get('3h', 0)
                    })

            if selected_forecast:
                st.success(f"✅ Weather forecast for {location_label} ({location_city}) on {selected_date}:")
                st.table(selected_forecast)
            else:
                st.warning("⚠️ No forecast data available for this date. (OpenWeather supports ~5 days ahead only).")

    except Exception as e:
        st.error(f"❌ Error fetching weather data: {e}")
