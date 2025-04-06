import os
import fastf1
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from datetime import datetime
import pytz

# Set up cache
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Streamlit app
st.title("F1 FastF1 App: Explore Race Results + Predict Next Race")

st.write("Use the dropdown menus below to select a season and round number to display race results.")

# Manual year and round input as dropdown menus
year_options = list(range(2018, 2025 + 1))
round_options = list(range(1, 24 + 1))

year = st.selectbox("Select Year", options=year_options, index=year_options.index(2023))
round_number = st.selectbox("Select Round", options=round_options, index=0)

# Declare session globally
session = None

# --- Existing manual race loader ---
if st.button("Load Race Data"):
    try:
        st.write(f"Loading race data for **Year {year}, Round {round_number}**...")

        session = fastf1.get_session(year, round_number, 'R')
        session.load()

        st.success(f"✅ Successfully loaded: {session.event['EventName']} ({session.date.date()})")

        # Show session info
        st.write("### Session Info:")
        st.json(session.event.to_dict())

        # Show race results
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

# --- New: Predict next race ---
if st.button("Generate Next Race Predictions"):
    try:
        st.write("Preparing data for prediction...")

        # Prepare historical training data
        historical_data = []
        le_driver = LabelEncoder()
        le_team = LabelEncoder()
        drivers_list = []
        teams_list = []

        for past_year in range(2018, 2024):
            for past_round in range(1, 24):
                try:
                    past_session = fastf1.get_session(past_year, past_round, 'R')
                    past_session.load()

                    if past_session.results is None or past_session.results.empty:
                        continue

                    df = past_session.results.copy()
                    df['win'] = (df['Position'] == 1).astype(int)

                    drivers_list.extend(df['Abbreviation'].unique())
                    teams_list.extend(df['TeamName'].unique())

                    df['grid_position'] = df['GridPosition']
                    df['points'] = df['Points']
                    df['laps'] = 0

                    df['year'] = past_year
                    df['round'] = past_round

                    historical_data.append(df)

                except Exception:
                    continue

        if not historical_data:
            st.warning("No historical data available to train the model.")
            st.stop()

        df_historical = pd.concat(historical_data)

        # Fit encoders
        le_driver.fit(drivers_list)
        le_team.fit(teams_list)

        df_historical['driver_encoded'] = le_driver.transform(df_historical['Abbreviation'])
        df_historical['team_encoded'] = le_team.transform(df_historical['TeamName'])

        features = ['grid_position', 'team_encoded', 'points', 'laps']
        X = df_historical[features]
        y = df_historical['win']

        # Train model
        base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        calibrated_model.fit(X, y)

        # Prepare next race data
        upcoming_year = 2024
        next_round = 1

        # Find the next scheduled race
        found_next_race = False
        for round_num in range(1, 24):
            try:
                future_session = fastf1.get_session(upcoming_year, round_num, 'R')
                future_session.load()

                if future_session.results is not None and not future_session.results.empty:
                    continue  # Skip completed races

                next_round = round_num
                found_next_race = True
                break
            except Exception:
                continue

        if not found_next_race:
            st.warning("No upcoming race found in the schedule.")
            st.stop()

        # Display race info
        session = fastf1.get_session(upcoming_year, next_round, 'R')
        session.load()

        # Timezone conversion
        utc_datetime = session.date
        nz_tz = pytz.timezone('Pacific/Auckland')
        nz_datetime = utc_datetime.astimezone(nz_tz)

        st.success(f"Next race: **{session.event['EventName']}** at **{session.event['Location']}**")
        st.info(f"Race date/time (NZST): **{nz_datetime.strftime('%Y-%m-%d %H:%M:%S')}**")

        if session.results is None or session.results.empty:
            st.warning("No entry list or results yet for this session.")
            st.stop()

        df_test = session.results.copy()

        df_test['grid_position'] = df_test['GridPosition']
        df_test['points'] = df_test['Points']
        df_test['laps'] = 0

        df_test['driver_encoded'] = le_driver.transform(df_test['Abbreviation'])
        df_test['team_encoded'] = le_team.transform(df_test['TeamName'])

        X_test = df_test[features]
        df_test['win_probability'] = calibrated_model.predict_proba(X_test)[:, 1]

        df_predictions = df_test[['Abbreviation', 'win_probability']].sort_values(by='win_probability', ascending=False)
        df_predictions.columns = ['Driver', 'Win Probability']

        st.write("### Win Probabilities for Next Race")
        st.dataframe(df_predictions)

    except Exception as e:
        st.error(f"❌ Error generating predictions: {e}")
