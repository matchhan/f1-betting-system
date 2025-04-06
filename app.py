import requests
import pandas as pd
import streamlit as st
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pytz
import joblib

# ------------------------- Helper Functions -------------------------

# Convert decimal odds to implied probability
def calculate_implied_probability(odds):
    try:
        return round(1 / odds, 4)
    except ZeroDivisionError:
        return 0.0

# Calculate stake using the Kelly Criterion
def kelly_fraction(probability, odds, fraction=0.125):
    edge = (odds * probability) - 1
    if edge <= 0:
        return 0
    return min(edge / (odds - 1) * fraction, 0.025)  # Cap at 2.5%

# Format bankroll output (NZD)
def format_currency(amount):
    return f"${amount:,.2f} NZD"

# Get current time in NZ timezone
def get_nz_time():
    return datetime.now(pytz.timezone("Pacific/Auckland")).strftime("%Y-%m-%d %H:%M:%S")

# ------------------------- Weather Data Collection -------------------------

def get_weather_data(api_key, circuit):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={circuit}&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    return data['main']['temp'], data['weather'][0]['description']

# ------------------------- Odds Collection from API -------------------------

# TheOddsAPI - API Key
API_KEY = st.secrets["odds_api"]["api_key"]
BASE_URL = "https://api.the-odds-api.com/v4/sports/motorsport_formula_one/odds"

def scrape_odds():
    try:
        params = {
            "apiKey": API_KEY,
            "regions": "eu",
            "markets": "h2h",
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
        return df

    except Exception as e:
        return pd.DataFrame(), f"Error scraping Pinnacle odds: {str(e)}"

# ------------------------- Machine Learning Model -------------------------

def train_model():
    df = scrape_odds()
    if df.empty:
        st.error("No data available for model training.")
        return None

    X = df[['implied_probability', 'odds']]  # Example features
    y = df['market']  # Predict market type (this would need adjustment)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Model trained with accuracy: {accuracy:.2f}")
    
    # Save model
    joblib.dump(model, 'f1_model.joblib')

    return model

# ------------------------- Betting & Notifications -------------------------

def calculate_bet_size(probability, odds, bankroll):
    stake = kelly_fraction(probability, odds)
    stake_amount = stake * bankroll
    return stake_amount

def send_telegram_notification(message):
    bot_token = st.secrets["telegram_bot_token"]
    chat_id = st.secrets["telegram_chat_id"]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url)

# ------------------------- Streamlit Interface -------------------------

# Page 1: Manual Bet Verification
def bet_verification_page():
    st.title("Manual Bet Verification")
    bet_time = st.text_input("Enter Race Time (YYYY-MM-DD HH:MM:SS)", "")
    current_odds = st.number_input("Enter Current Odds", min_value=1.0)
    bet_odds = st.number_input("Enter Bet Odds", min_value=1.0)
    probability = st.number_input("Enter Predicted Probability", min_value=0.0, max_value=1.0)

    if st.button("Verify Bet"):
        bet_size = calculate_bet_size(probability, bet_odds, 500)  # Example: using $500 bankroll
        message = f"Bet verification complete: \nRace Time: {bet_time}\nOdds: {current_odds}\nBet Size: {bet_size}"
        send_telegram_notification(message)

# Page 2: Model Training and Betting Notification
def model_training_page():
    st.title("Train Machine Learning Model")
    if st.button("Train Model"):
        model = train_model()
        if model:
            st.success("Model trained successfully!")

    df = scrape_odds()
    if not df.empty:
        st.dataframe(df)

# ------------------------- Main Streamlit App -------------------------

def main():
    st.sidebar.title("F1 Betting System")

    page = st.sidebar.selectbox("Select a Page", ["Bet Verification", "Model Training"])

    if page == "Bet Verification":
        bet_verification_page()
    elif page == "Model Training":
        model_training_page()

# Load your Telegram credentials from Streamlit secrets
telegram_bot_token = st.secrets["telegram_bot_token"]
telegram_chat_id = st.secrets["telegram_chat_id"]

# Function to send a test message to Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": telegram_chat_id,
        "text": message
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            st.success("Test message sent successfully!")
        else:
            st.error(f"Failed to send message: {response.text}")
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")

# Add a button in the Streamlit app to send the test message
if st.button('Send Test Telegram Message'):
    send_telegram_message("This is a test message from the F1 Betting System.")

if __name__ == "__main__":
    main()

