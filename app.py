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
                    # Fetch race result and check if the result is a valid dictionary
                    try:
                        race_result = fastf1.get_race_result(year, race['round'])
                        if not isinstance(race_result, dict):
                            st.error(f"Unexpected data type for race result: {type(race_result)}")
                            continue
                    except Exception as e:
                        st.error(f"Error fetching race result: {str(e)}")
                        continue

                    # Similarly, handle qualifying and practice session results
                    try:
                        qualifying = fastf1.get_qualifying_results(year, race['round'])
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
                        # Check the practice sessions to ensure they are valid
                        for session in practice_sessions:
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

def calculate_implied_probability(odds):
    try:
        return round(1 / odds, 4)
    except ZeroDivisionError:
        return 0.0

def format_scraper_log(df):
    if df.empty:
        return "No odds found."
    event_time = df['event_time'].iloc[0].strftime("%Y-%m-%d %H:%M:%S") if 'event_time' in df.columns and not df['event_time'].isna().all() else "Unknown"
    last_update = df['last_update'].iloc[0] if 'last_update' in df.columns and not df['last_update'].isna().all() else "Unknown"
    log = f"✅ Odds Scraped: {len(df)} entries\n🕒 Event Time: {event_time}\n🔄 Last Update from Bookmaker: {last_update}"
    return log

# Convert bankroll percentage to fixed stake amount
def calculate_stake(bankroll, percentage):
    return round(bankroll * (percentage / 100), 2)

# Kelly Criterion calculator (1/8 Kelly as you prefer)
def kelly_fraction(probability, odds, fraction=0.125):
    edge = (odds * probability) - 1
    if edge <= 0:
        return 0
    return min(edge / (odds - 1) * fraction, 0.025)

# Manually enter race data
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

# Scrape Betcha F1 Odds
def scrape_betcha_odds(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')

        all_odds = []
        
        race_events = soup.find_all("div", class_="race-event-class")  # Replace with correct class

        for race_event in race_events:
            race_name = race_event.find("div", class_="race-name").text.strip()  # Update class
            event_time = race_event.find("div", class_="event-time").text.strip()  # Update class
            drivers = race_event.find_all("div", class_="driver-class")  # Update class for drivers

            for driver in drivers:
                driver_name = driver.find("span", class_="driver-name").text.strip()  # Update class
                odds = float(driver.find("span", class_="odds-amount").text.strip())  # Update class
                implied_prob = calculate_implied_probability(odds)

                all_odds.append({
                    "sport": "F1",
                    "race": race_name,
                    "event_time": event_time,
                    "driver": driver_name,
                    "bookmaker": "Betcha",
                    "market": "h2h",  # Head-to-head market
                    "odds": odds,
                    "implied_probability": implied_prob,
                })

        df = pd.DataFrame(all_odds)

        if df.empty:
            return pd.DataFrame(), "No Betcha F1 odds found."

        return df, format_scraper_log(df)

    except Exception as e:
        return pd.DataFrame(), f"Error scraping Betcha odds: {str(e)}"

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
