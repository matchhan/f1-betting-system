import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import pytz
import joblib
import telegram
import os
import fastf1
import asyncio  # Import asyncio to handle async calls

# Telegram Bot Setup
telegram_bot_token = "7630284552:AAHKzWIuRqMCon032ycjHC5r2AME-y-JEho"
telegram_chat_id = "7863097165"

# The Odds API Setup
API_KEY = "475a918d769547f547ae7dbb6ddf02a8"
BASE_URL = "https://api.the-odds-api.com/v4/sports/motorsport_formula_one/odds"

# Weather API Setup
weather_api_key = "9902ad598ba458f05379b0deb1f086b7"
weather_base_url = "https://api.openweathermap.org/data/2.5/weather"

# Function to calculate implied probability from odds
def calculate_implied_probability(odds):
    try:
        return round(1 / odds, 4)
    except ZeroDivisionError:
        return 0.0

# Load and preprocess F1 Data using FastF1
def load_data():
    if os.path.exists("f1_data.csv"):
        return pd.read_csv('f1_data.csv')
    else:
        st.write("No data found. Fetching new data...")
        data = fetch_f1_data()
        return data

def fetch_f1_data():
    try:
        cache_dir = 'f1_cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        fastf1.Cache.enable_cache(cache_dir)  # Enable caching to store F1 data

        years = range(2018, 2023)
        race_data = []

        for year in years:
            season = fastf1.get_event_schedule(year)
            for race in season:
                if race['raceName'] not in ['Sprint', 'Qualifying']:  # Skip sprint races
                    try:
                        race_result = fastf1.get_race_result(year, race['round'])
                        st.write(f"Race result for {race['raceName']} (Year {year}, Round {race['round']}): {race_result}")  # Debug output
                        if not isinstance(race_result, dict):
                            st.error(f"Unexpected data type for race result: {type(race_result)}")
                            continue
                    except Exception as e:
                        st.error(f"Error fetching race result: {str(e)}")
                        continue

                    try:
                        qualifying = fastf1.get_qualifying_results(year, race['round'])
                        st.write(f"Qualifying data for {race['raceName']} (Year {year}, Round {race['round']}): {qualifying}")  # Debug output
                        if not isinstance(qualifying, dict):
                            st.error(f"Unexpected data type for qualifying: {type(qualifying)}")
                            continue
                    except Exception as e:
                        st.error(f"Error fetching qualifying results: {str(e)}")
                        continue

                    try:
                        practice_sessions = [
                            fastf1.get_practice_data(year, race['round'], 1),
                            fastf1.get_practice_data(year, race['round'], 2),
                            fastf1.get_practice_data(year, race['round'], 3),
                        ]
                        for session in practice_sessions:
                            st.write(f"Practice session data: {session}")  # Debug output
                            if not isinstance(session, dict):
                                st.error(f"Unexpected data type for practice session: {type(session)}")
                                continue
                    except Exception as e:
                        st.error(f"Error fetching practice data: {str(e)}")
                        continue
                    
                    # Handle weather data
                    weather = fetch_weather_data(race['location']['locality'])

                    # Process the race result data
                    for driver in race_result['results']:
                        driver_name = driver['Driver']['familyName']
                        finishing_position = driver['positionOrder']
                        time = driver['Time']['time'] if 'Time' in driver else "N/A"
                        lap_time = driver['FastestLap']['Time']['time'] if 'FastestLap' in driver else "N/A"

                        race_data.append({
                            "year": year,
                            "round": race['round'],
                            "race_name": race['raceName'],
                            "driver": driver_name,
                            "finish_position": finishing_position,
                            "time": time,
                            "lap_time": lap_time,
                            "weather": weather,
                        })

        df = pd.DataFrame(race_data)
        df.to_csv("f1_data.csv", index=False)  # Save the data to CSV for future use
        st.write("F1 data fetched and saved!")
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def fetch_weather_data(city_name):
    try:
        params = {'q': city_name, 'appid': weather_api_key, 'units': 'metric'}
        response = requests.get(weather_base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        temperature = weather_data['main']['temp']
        weather_desc = weather_data['weather'][0]['description']
        return f"{temperature}°C, {weather_desc}"
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return "Unknown"

# Sending test telegram message
async def send_telegram_message(message):
    bot = telegram.Bot(token=telegram_bot_token)
    await bot.send_message(chat_id=telegram_chat_id, text=message)
    st.write("Test message sent to Telegram!")

def manual_data_entry():
    st.subheader("Manually Enter F1 Race Data")

    race_name = st.text_input("Race Name", "")
    event_time = st.date_input("Event Date", datetime.today())
    driver_name = st.text_input("Driver Name", "")
    odds = st.number_input("Odds", min_value=1.0, format="%.2f")
    
    implied_prob = calculate_implied_probability(odds)

    if st.button("Add Data"):
        data = {
            "sport": "F1",
            "race": race_name,
            "event_time": event_time,
            "driver": driver_name,
            "odds": odds,
            "implied_probability": implied_prob,
            "bookmaker": "Manual Entry",
            "market": "h2h",  # Assuming it's always a head-to-head market for now
        }

        df = pd.DataFrame([data])
        return df
    return None

# Main function with test buttons and logs
def main():
    st.title("F1 Betting System")

    # Load F1 data
    data = load_data()
    if data is not None:
        st.write("Displaying F1 Data:")
        st.dataframe(data)
    
    # Button to test loading F1 data
    if st.button("Test Fetch F1 Data"):
        st.write("Testing F1 data fetch...")
        test_data = fetch_f1_data()
        if test_data is not None:
            st.write("F1 Data fetched successfully!")
            st.dataframe(test_data)
        else:
            st.write("Error fetching F1 data.")
    
    # Button to send test Telegram message
    if st.button("Send Test Telegram"):
        asyncio.run(send_telegram_message("Test message from your F1 Betting System!"))
    
    # Radio button for data entry method
    entry_choice = st.radio("Choose Data Entry Method", ("Manually Enter Data", "Scrape Betcha Odds"))

    if entry_choice == "Manually Enter Data":
        df = manual_data_entry()
        if df is not None:
            st.write("Manually Entered Data:")
            st.dataframe(df)

    elif entry_choice == "Scrape Betcha Odds":
        url_input = st.text_input("Enter the URL for F1 Grand Prix Odds", "https://www.betcha.co.nz/sports/motor-sport/formula-1/gp-japan-race/768194d2-d4ec-4e32-b1da-b42a55116bc5")
        if st.button("Scrape Betcha Odds"):
            df, log = scrape_betcha_odds(url_input)
            st.write(log)
            if not df.empty:
                st.dataframe(df)
            else:
                st.write("No data available.")

if __name__ == "__main__":
    main()
