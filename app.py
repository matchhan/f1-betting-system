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

# Load and preprocess F1 Data using FastF1
def load_data():
    # Try to load the data from a stored CSV file first
    if os.path.exists("f1_data.csv"):
        return pd.read_csv('f1_data.csv')
    else:
        st.write("No data found. Fetching new data...")
        # Fetch new data if CSV file does not exist
        data = fetch_f1_data()
        return data

def fetch_f1_data():
    try:
        # Create the cache directory if it doesn't exist
        cache_dir = 'f1_cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Initialize FastF1
        fastf1.Cache.enable_cache(cache_dir)  # Enable caching to store F1 data

        # Define the race seasons and rounds to fetch data
        years = range(2018, 2023)  # Adjust years accordingly
        race_data = []

        for year in years:
            season = fastf1.get_event_schedule(year)
            for race in season:
                if race['raceName'] not in ['Sprint', 'Qualifying']:  # Skip sprint races
                    race_result = fastf1.get_race_result(year, race['round'])
                    qualifying = fastf1.get_qualifying_results(year, race['round'])
                    practice_sessions = [
                        fastf1.get_practice_data(year, race['round'], 1),  # Practice 1
                        fastf1.get_practice_data(year, race['round'], 2),  # Practice 2
                        fastf1.get_practice_data(year, race['round'], 3),  # Practice 3
                    ]

                    # Get weather data for the race location (city)
                    weather = fetch_weather_data(race['location']['locality'])

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
                            "weather": weather,  # Adding weather data to the race record
                        })

        # Convert race data to DataFrame
        df = pd.DataFrame(race_data)
        df.to_csv("f1_data.csv", index=False)  # Save the data to CSV for future use
        st.write("F1 data fetched and saved!")
        return df
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def fetch_weather_data(city_name):
    try:
        # Make the API request to OpenWeather
        params = {
            'q': city_name,
            'appid': weather_api_key,
            'units': 'metric'
        }
        response = requests.get(weather_base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()

        # Extract relevant weather data (temperature, weather description)
        temperature = weather_data['main']['temp']
        weather_desc = weather_data['weather'][0]['description']
        return f"{temperature}°C, {weather_desc}"

    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return "Unknown"

# Convert decimal odds to implied probability
def calculate_implied_probability(odds):
    try:
        return round(1 / odds, 4)
    except ZeroDivisionError:
        return 0.0

# Format scraper log for display
def format_scraper_log(df):
    if df.empty:
        return "No odds found."

    event_time = df['event_time'].iloc[0].strftime("%Y-%m-%d %H:%M:%S") if 'event_time' in df.columns and not df['event_time'].isna().all() else "Unknown"
    last_update = df['last_update'].iloc[0] if 'last_update' in df.columns and not df['last_update'].isna().all() else "Unknown"

    log = (
        f"✅ Odds Scraped: {len(df)} entries\n"
        f"🕒 Event Time: {event_time}\n"
        f"🔄 Last Update from Bookmaker: {last_update}"
    )
    return log

# Convert bankroll percentage to fixed stake amount
def calculate_stake(bankroll, percentage):
    return round(bankroll * (percentage / 100), 2)

# Get current time in NZ timezone
def get_nz_time():
    return datetime.now(pytz.timezone("Pacific/Auckland")).strftime("%Y-%m-%d %H:%M:%S")

# Kelly Criterion calculator (1/8 Kelly as you prefer)
def kelly_fraction(probability, odds, fraction=0.125):
    edge = (odds * probability) - 1
    if edge <= 0:
        return 0
    return min(edge / (odds - 1) * fraction, 0.025)  # Cap at 2.5% as you specified

# Format bankroll currency output (NZD)
def format_currency(amount):
    return f"${amount:,.2f} NZD"

# Train Machine Learning Model
def train_model(df):
    st.write("Training the model...")

    # Preprocess the data here (e.g., feature engineering)
    X = df.drop(columns=['race_result'])  # Replace with the correct features
    y = df['race_result']  # Replace with the correct target variable
    
    model = xgb.XGBClassifier()
    model.fit(X, y)

    joblib.dump(model, 'f1_model.joblib')
    st.write("Model training complete!")

# Send a test message to Telegram (make this async)
async def send_telegram_message(message):
    bot = telegram.Bot(token=telegram_bot_token)
    await bot.send_message(chat_id=telegram_chat_id, text=message)
    st.write("Test message sent to Telegram!")

# Scrape Pinnacle Odds
def scrape_pinnacle_odds():
    try:
        params = {
            "apiKey": API_KEY,
            "regions": "eu",
            "markets": "h2h",  # Head to head = race winner
            "oddsFormat": "decimal",
            "bookmakers": "pinnacle"
        }

        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        all_odds = []

        for event in data:
            race_name = event.get("home_team", "Unknown Race")
            commence_time = event.get("commence_time", None)
            event_time = datetime.fromisoformat(commence_time[:-1]) if commence_time else None

            for bookmaker in event.get("bookmakers", []):
                if bookmaker["key"] != "pinnacle":
                    continue

                for market in bookmaker.get("markets", []):
                    if market["key"] != "h2h":
                        continue

                    for outcome in market.get("outcomes", []):
                        driver = outcome["name"]
                        odds = outcome["price"]
                        implied_prob = calculate_implied_probability(odds)

                        all_odds.append({
                            "sport": "F1",
                            "race": race_name,
                            "event_time": event_time,
                            "driver": driver,
                            "bookmaker": bookmaker["title"],
                            "market": market["key"],
                            "odds": odds,
                            "implied_probability": implied_prob,
                            "last_update": bookmaker["last_update"]
                        })

        df = pd.DataFrame(all_odds)

        if df.empty:
            return pd.DataFrame(), "No Pinnacle F1 odds found."

        return df, format_scraper_log(df)

    except Exception as e:
        return pd.DataFrame(), f"Error scraping Pinnacle odds: {str(e)}"

# Main function
def main():
    st.title("F1 Betting System")

    # Load F1 data
    data = load_data()
    if data is not None:
        st.write("Displaying F1 Data:")
        st.dataframe(data)
    
    # Train model button
    if st.button("Train Machine Learning Model"):
        if data is not None:
            train_model(data)
    
    # Send test Telegram button
    if st.button("Send Test Telegram"):
        asyncio.run(send_telegram_message("Test message from your F1 Betting System!"))
    
    # Scrape Pinnacle Odds button
    if st.button("Scrape Pinnacle Odds"):
        df, log = scrape_pinnacle_odds()
        st.write(log)
        if not df.empty:
            st.dataframe(df)
        else:
            st.write("No data available.")

if __name__ == "__main__":
    main()
