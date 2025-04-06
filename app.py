import os
import fastf1
import pandas as pd
import streamlit as st
from datetime import timedelta, timezone
import pytz

# Machine learning libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

# Set up FastF1 cache
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Streamlit app
st.title("F1 FastF1 Test App: Explore Race Results & Predictions")

st.write("Use the dropdown menus below to select a season and round number to display race results.")

# Manual year and round input as dropdown menus
year_options = list(range(2018, 2025 + 1))
round_options = list(range(1, 24 + 1))

year = st.selectbox("Select Year", options=year_options, index=year_options.index(2023))
round_number = st.selectbox("Select Round", options=round_options, index=0)

if st.button("Load Race Data"):
    try:
        st.write(f"Loading race data for **Year {year}, Round {round_number}**...")

        session = fastf1.get_session(year, round_number, 'R')
        session.load()

        st.success(f"✅ Successfully loaded: {session.event['EventName']} ({session.date.date()})")

        # Show session info
        st.write("### Session Info:")
        st.json(session.event.to_dict())

        # Show race results in table
        results = session.results
        if results is not None and not results.empty:
            df_results = results[['Abbreviation', 'Position', 'TeamName', 'Points']].sort_values(by='Position')
            df_results.columns = ['Driver', 'Finish Position', 'Team', 'Points']
            st.write("### Final Race Results")
            st.dataframe(df_results)
        else:
            st.warning("No results available for this session.")

        # Show lap data sample
        if not session.laps.empty:
            st.write("### Sample Lap Data")
            st.dataframe(session.laps.head())
        else:
            st.warning("No lap data available.")

    except Exception as e:
        st.error(f"❌ Error loading session: {e}")

# =========================
# Machine Learning Section
# =========================

@st.cache_data
def prepare_training_data():
    data = []

    for year in range(2018, 2024):  # use past seasons
        try:
            for round_number in range(1, 23):
                session = fastf1.get_session(year, round_number, 'R')
                session.load()

                if session.results is None or session.results.empty:
                    continue

                for _, row in session.results.iterrows():
                    data.append({
                        'year': year,
                        'round': round_number,
                        'driver': row['Abbreviation'],
                        'team': row['TeamName'],
                        'grid_position': row['GridPosition'],
                        'finish_position': row['Position'],
                        'points': row['Points'],
                        'status': row['Status'],
                    })
        except Exception:
            continue

    df = pd.DataFrame(data)
    df.dropna(inplace=True)

    # Encode categorical variables
    le_driver = LabelEncoder()
    le_team = LabelEncoder()

    df['driver_encoded'] = le_driver.fit_transform(df['driver'])
    df['team_encoded'] = le_team.fit_transform(df['team'])
    df['target'] = (df['finish_position'] == 1).astype(int)  # Predict win: 1 if win, 0 otherwise

    features = ['grid_position', 'driver_encoded', 'team_encoded']
    target = 'target'

    return df[features], df[target], le_driver, le_team

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_test, y_test)

    return calibrated_model

def predict_next_race(model, le_driver, le_team):
    # Find next race
    schedule = fastf1.get_event_schedule(2024, include_testing=False)

    # Get the next race (assumes future races are marked as such)
    upcoming_races = schedule[schedule['Session5Date'] > pd.Timestamp.now(tz=timezone.utc)]

    if upcoming_races.empty:
        return None, None, None

    next_race = upcoming_races.iloc[0]
    round_number = next_race['RoundNumber']
    location = next_race['Location']
    event_name = next_race['EventName']
    event_date_utc = next_race['Session5Date']

    # Convert time to NZST
    nz_tz = pytz.timezone("Pacific/Auckland")
    event_date_nz = event_date_utc.tz_convert(nz_tz)

    # Load latest data (qualifying/grid position may not be ready, fallback to last race)
    try:
        session = fastf1.get_session(2024, round_number, 'R')
        session.load()
    except Exception:
        return None, event_name, event_date_nz

    if session.results.empty:
        return None, event_name, event_date_nz

    input_data = []
    drivers = []

    for _, row in session.results.iterrows():
        drivers.append(row['Abbreviation'])

        try:
            driver_encoded = le_driver.transform([row['Abbreviation']])[0]
        except:
            driver_encoded = -1  # Handle unseen drivers

        try:
            team_encoded = le_team.transform([row['TeamName']])[0]
        except:
            team_encoded = -1  # Handle unseen teams

        input_data.append([
            row['GridPosition'],
            driver_encoded,
            team_encoded,
        ])

    probabilities = model.predict_proba(input_data)[:, 1]  # Probability of winning

    df_predictions = pd.DataFrame({
        'Driver': drivers,
        'Win Probability': probabilities
    }).sort_values(by='Win Probability', ascending=False)

    return df_predictions, event_name, event_date_nz

# Streamlit Button for Model Training and Prediction
if st.button("Train Model and Predict Next Race"):
    with st.spinner("Training model..."):
        X, y, le_driver, le_team = prepare_training_data()
        model = train_model(X, y)

    st.success("Model trained!")

    with st.spinner("Generating predictions for next race..."):
        predictions, race_name, race_time_nz = predict_next_race(model, le_driver, le_team)

    if predictions is not None:
        st.write(f"### Predictions for Next Race: {race_name}")
        st.write(f"📍 **Race Time (NZST):** {race_time_nz.strftime('%Y-%m-%d %H:%M:%S')}")
        st.dataframe(predictions)
    else:
        st.warning("Could not generate predictions — next race data unavailable or incomplete.")
