import fastf1
import pandas as pd
import numpy as np
import streamlit as st
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb

# Initialise FastF1 cache
fastf1.Cache.enable_cache('./fastf1_cache')

# Streamlit App
st.title("F1 Race Win Probability Predictor")
st.write("Click the button below to generate win probabilities for the next race.")

if st.button('Predict Next Race Probabilities'):

    # Step 1: Data collection
    years = list(range(2018, datetime.now().year + 1))
    all_race_data = []

    st.write("Collecting race data...")

    for year in tqdm(years, desc="Seasons"):
        schedule = fastf1.get_event_schedule(year)

        for _, race in schedule.iterrows():
            race_name = race['EventName']
            race_date = race['Session1Date'].date()
            round_number = race['RoundNumber']

            try:
                session = fastf1.get_session(year, round_number, 'R')
                session.load()
            except Exception as e:
                print(f"Failed to load session {race_name}: {e}")
                continue

            results = session.results
            if results is None:
                continue

            for index, row in results.iterrows():
                all_race_data.append({
                    'year': year,
                    'round': round_number,
                    'race_name': race_name,
                    'date': race_date,
                    'driver': row['Abbreviation'],
                    'team': row['TeamName'],
                    'grid_position': row['GridPosition'],
                    'position': row['Position'],
                    'points': row['Points'],
                    'laps': row['Laps'],
                    'status': row['Status'],
                    'fastest_lap_time': row['FastestLapTime'],
                    'fastest_lap_speed': row['FastestLapSpeed'],
                })

    df = pd.DataFrame(all_race_data)

    if df.empty:
        st.error("No race data collected. Please check FastF1 installation.")
    else:
        st.success(f"Collected data for {df.shape[0]} race entries.")

        # Step 2: Preprocess data
        st.write("Preprocessing data...")

        df['win'] = (df['position'] == 1).astype(int)

        le_driver = LabelEncoder()
        le_team = LabelEncoder()

        df['driver_encoded'] = le_driver.fit_transform(df['driver'])
        df['team_encoded'] = le_team.fit_transform(df['team'])

        features = ['grid_position', 'team_encoded', 'points', 'laps', 'fastest_lap_speed']
        df = df.dropna(subset=features)

        X = df[features]
        y = df['win']

        # Step 3: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Model training with calibration
        st.write("Training model...")
        base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)

        # Step 5: Evaluate model
        y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
        y_pred = calibrated_model.predict(X_test)

        st.write(f'Log Loss: {log_loss(y_test, y_pred_proba):.4f}')
        st.write(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

        # Step 6: Predict next race
        st.write("Predicting next race...")

        latest_year = datetime.now().year
        schedule = fastf1.get_event_schedule(latest_year)
        upcoming_races = schedule[schedule['Session1Date'] > datetime.now()]

        if upcoming_races.empty:
            st.warning("No upcoming races found.")
        else:
            next_race = upcoming_races.iloc[0]
            round_number = next_race['RoundNumber']

            try:
                session = fastf1.get_session(latest_year, round_number, 'R')
                session.load()

                race_data = []

                results = session.results
                if results is None:
                    st.warning("No results available yet for the next race.")
                else:
                    for index, row in results.iterrows():
                        race_data.append({
                            'grid_position': row['GridPosition'],
                            'team_encoded': le_team.transform([row['TeamName']])[0] if row['TeamName'] in le_team.classes_ else 0,
                            'points': row['Points'],
                            'laps': row['Laps'],
                            'fastest_lap_speed': row['FastestLapSpeed'],
                            'driver': row['Abbreviation']
                        })

                    upcoming_df = pd.DataFrame(race_data)
                    probabilities = calibrated_model.predict_proba(upcoming_df[features])[:, 1]
                    upcoming_df['win_probability'] = probabilities

                    st.write("Win Probabilities for Next Race:")
                    st.dataframe(upcoming_df[['driver', 'win_probability']].sort_values(by='win_probability', ascending=False))

            except Exception as e:
                st.error(f"Failed to load next race session: {e}")
