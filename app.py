import os
import logging
import fastf1
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb

# Suppress all FastF1 internal logs and urllib3
logging.getLogger('fastf1').setLevel(logging.CRITICAL)
logging.getLogger('fastf1.api').setLevel(logging.CRITICAL)
logging.getLogger('fastf1.req').setLevel(logging.CRITICAL)
logging.getLogger('fastf1.events').setLevel(logging.CRITICAL)
logging.getLogger('fastf1.livetiming').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

# Ensure FastF1 cache directory exists
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)

# Initialise FastF1 cache
fastf1.Cache.enable_cache(cache_dir)

# Streamlit App
st.title("F1 Race Win Probability Predictor")
st.write("Click the button below to generate win probabilities for the next race.")
st.write("Or check the last completed race to confirm data is up to date.")

# --- New: Check Previous Race Results Button ---
if st.button('Show Previous Race Results'):
    latest_year = datetime.now().year
    found_previous_race = False

    for round_number in reversed(range(1, 25)):
        try:
            session = fastf1.get_session(latest_year, round_number, 'R')
            session.load()

            if not session.loaded:
                continue

            results = session.results
            if results is None or results.empty:
                continue

            st.success(f"Previous race: Round {round_number} - {session.event.get('EventName', '')}")
            df_results = results[['Abbreviation', 'Position', 'TeamName', 'Points']].sort_values(by='Position')
            st.dataframe(df_results.rename(columns={
                'Abbreviation': 'Driver',
                'TeamName': 'Team',
                'Position': 'Finish Position',
                'Points': 'Points Earned'
            }))

            found_previous_race = True
            break

        except Exception:
            continue

    if not found_previous_race:
        st.warning("No completed race found yet for this season.")

# --- Predict Next Race Probabilities ---
if st.button('Predict Next Race Probabilities'):

    # Step 1: Data collection
    st.write("Collecting race data...")

    years = list(range(2018, datetime.now().year + 1))  # All years up to current year
    all_race_data = []

    for year in years:
        for round_number in range(1, 25):  # Up to 24 rounds per season
            try:
                session = fastf1.get_session(year, round_number, 'R')
                session.load()

                if not session.loaded:
                    continue

                race_name = session.event.get('EventName', f"Round {round_number}")
                race_date = session.date.date() if session.date else None

            except Exception:
                continue

            results = session.results
            if results is None or results.empty:
                continue

            if not session.laps.empty:
                laps_completed = session.laps.groupby('Driver')['LapNumber'].max().reset_index()
                laps_completed.rename(columns={'LapNumber': 'laps'}, inplace=True)
            else:
                laps_completed = pd.DataFrame({'Driver': results['Abbreviation'], 'laps': np.nan})

            results = results.merge(laps_completed, left_on='Abbreviation', right_on='Driver', how='left')
            results.drop(columns=['Driver'], inplace=True)

            for index, row in results.iterrows():
                all_race_data.append({
                    'year': year,
                    'round': round_number,
                    'race_name': race_name,
                    'date': race_date,
                    'driver': row.get('Abbreviation', np.nan),
                    'team': row.get('TeamName', np.nan),
                    'grid_position': row.get('GridPosition', np.nan),
                    'position': row.get('Position', np.nan),
                    'points': row.get('Points', np.nan),
                    'laps': row.get('laps', np.nan),
                    'status': row.get('Status', np.nan),
                    'fastest_lap_time': row.get('FastestLapTime', np.nan),
                    'fastest_lap_speed': row.get('FastestLapSpeed', np.nan),
                })

    df = pd.DataFrame(all_race_data)

    if df.empty:
        st.error("No race data collected. Please check FastF1 installation or API availability.")
    else:
        st.success(f"Collected data for {df.shape[0]} race entries.")

        st.write("Data Collection Summary:")
        st.write(f"- Total race entries: {df.shape[0]}")
        st.write(f"- Missing lap counts: {df['laps'].isna().sum()}")
        st.write(f"- Missing fastest lap times: {df['fastest_lap_time'].isna().sum()}")
        st.write(f"- Missing fastest lap speeds: {df['fastest_lap_speed'].isna().sum()}")

        # Step 2: Preprocess data
        st.write("Preprocessing data...")

        df['win'] = (df['position'] == 1).astype(int)

        le_driver = LabelEncoder()
        le_team = LabelEncoder()

        df['driver_encoded'] = le_driver.fit_transform(df['driver'])
        df['team_encoded'] = le_team.fit_transform(df['team'])

        df.sort_values(by=['driver', 'year', 'round'], inplace=True)
        df['race_number'] = df.groupby('driver').cumcount() + 1

        df['avg_position_last_3'] = df.groupby('driver')['position'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        df['avg_position_last_5'] = df.groupby('driver')['position'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        st.write("Added average positions for last 3 and 5 races.")

        features = ['grid_position', 'team_encoded', 'points', 'laps', 'fastest_lap_speed',
                    'avg_position_last_3', 'avg_position_last_5']
        df = df.dropna(subset=features)

        X = df[features]
        y = df['win']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write("Training model...")
        base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)

        y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
        y_pred = calibrated_model.predict(X_test)

        st.write(f'Log Loss: {log_loss(y_test, y_pred_proba):.4f}')
        st.write(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

        st.write("Predicting next race...")

        latest_year = datetime.now().year
        for round_number in range(1, 25):
            try:
                session = fastf1.get_session(latest_year, round_number, 'R')
                session.load()

                if not session.loaded:
                    continue

                results = session.results
                if results is None or results.empty:
                    continue

                st.success(f"Next race found: Round {round_number} {session.event.get('EventName', '')}")
                race_data = []
                history = df.copy()

                for index, row in results.iterrows():
                    driver = row.get('Abbreviation', np.nan)
                    driver_history = history[history['driver'] == driver].tail(5)

                    avg_pos_last_3 = driver_history.tail(3)['position'].mean() if not driver_history.tail(3).empty else np.nan
                    avg_pos_last_5 = driver_history['position'].mean() if not driver_history.empty else np.nan

                    race_data.append({
                        'grid_position': row.get('GridPosition', np.nan),
                        'team_encoded': le_team.transform([row.get('TeamName', '')])[0] if row.get('TeamName', '') in le_team.classes_ else 0,
                        'points': row.get('Points', np.nan),
                        'laps': row.get('Laps', np.nan),
                        'fastest_lap_speed': row.get('FastestLapSpeed', np.nan),
                        'avg_position_last_3': avg_pos_last_3,
                        'avg_position_last_5': avg_pos_last_5,
                        'driver': driver
                    })

                upcoming_df = pd.DataFrame(race_data).dropna(subset=features)

                if upcoming_df.empty:
                    st.warning("Not enough data to make predictions for the next race.")
                else:
                    probabilities = calibrated_model.predict_proba(upcoming_df[features])[:, 1]
                    upcoming_df['win_probability'] = probabilities

                    st.write("Win Probabilities for Next Race:")
                    st.dataframe(upcoming_df[['driver', 'win_probability']].sort_values(by='win_probability', ascending=False))

                break  # Stop after finding the first future race

            except Exception:
                continue
