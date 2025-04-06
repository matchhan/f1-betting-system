import os
import fastf1
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

# Set up cache
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Streamlit app
st.title("F1 Win Probability Predictor")

st.write("Use the dropdown menus below to select a season and round number to predict win probabilities.")

# Manual year and round input as dropdown menus
year_options = list(range(2018, datetime.now().year + 1))
round_options = list(range(1, 24 + 1))

year = st.selectbox("Select Year", options=year_options, index=year_options.index(datetime.now().year))
round_number = st.selectbox("Select Round", options=round_options, index=0)

if st.button("Generate Predictions"):

    try:
        st.write("Loading historical race data...")

        # Step 1: Collect historical race data
        years = list(range(2018, datetime.now().year))
        all_race_data = []

        for y in years:
            try:
                schedule = fastf1.get_event_schedule(y)
            except Exception:
                continue  # Skip if schedule cannot be loaded

            for _, race in schedule.iterrows():
                round_no = race['RoundNumber']

                try:
                    session = fastf1.get_session(y, round_no, 'R')
                    session.load()
                except Exception:
                    continue  # Skip session if it fails to load

                if session.results is None:
                    continue

                results = session.results.copy()

                # Calculate laps completed
                if not session.laps.empty:
                    laps_completed = session.laps.groupby('Driver')['LapNumber'].max().reset_index()
                    laps_completed.rename(columns={'LapNumber': 'laps'}, inplace=True)
                    results = results.merge(laps_completed, left_on='Abbreviation', right_on='Driver', how='left')
                    results.drop(columns=['Driver'], inplace=True)
                else:
                    results['laps'] = np.nan

                for _, row in results.iterrows():
                    all_race_data.append({
                        'year': y,
                        'round': round_no,
                        'driver': row['Abbreviation'],
                        'team': row['TeamName'],
                        'grid_position': row['GridPosition'],
                        'position': row['Position'],
                        'points': row['Points'],
                        'laps': row['laps'],
                    })

        df = pd.DataFrame(all_race_data)

        if df.empty:
            st.error("No race data collected.")
        else:
            st.write(f"Collected data for {df.shape[0]} race entries.")

            # Step 2: Preprocess data
            df['win'] = (df['position'] == 1).astype(int)

            le_driver = LabelEncoder()
            le_team = LabelEncoder()

            df['driver_encoded'] = le_driver.fit_transform(df['driver'])
            df['team_encoded'] = le_team.fit_transform(df['team'])

            # Recency features
            df.sort_values(by=['driver', 'year', 'round'], inplace=True)
            df['avg_position_last_3'] = df.groupby('driver')['position'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            df['avg_position_last_5'] = df.groupby('driver')['position'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

            # Drop rows with missing values in selected features
            features = ['grid_position', 'team_encoded', 'points', 'laps', 'avg_position_last_3', 'avg_position_last_5']
            df = df.dropna(subset=features)

            # Step 3: Train model
            X = df[features]
            y = df['win']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            calibrated_model.fit(X_train, y_train)

            st.success("Model training complete.")

            # Step 4: Predict selected race
            try:
                session = fastf1.get_session(year, round_number, 'R')
                session.load()

                if session.results is None:
                    st.warning("No results available yet for the selected race (grid not set).")
                else:
                    race_data = []
                    history = df.copy()

                    for _, row in session.results.iterrows():
                        driver = row['Abbreviation']
                        driver_history = history[history['driver'] == driver].tail(5)

                        avg_pos_last_3 = driver_history.tail(3)['position'].mean() if not driver_history.tail(3).empty else np.nan
                        avg_pos_last_5 = driver_history['position'].mean() if not driver_history.empty else np.nan

                        race_data.append({
                            'grid_position': row['GridPosition'],
                            'team_encoded': le_team.transform([row['TeamName']])[0] if row['TeamName'] in le_team.classes_ else 0,
                            'points': row['Points'],
                            'laps': row['Laps'],
                            'avg_position_last_3': avg_pos_last_3,
                            'avg_position_last_5': avg_pos_last_5,
                            'driver': driver
                        })

                    upcoming_df = pd.DataFrame(race_data).dropna(subset=features)

                    if upcoming_df.empty:
                        st.warning("Not enough data to make predictions for this race.")
                    else:
                        probabilities = calibrated_model.predict_proba(upcoming_df[features])[:, 1]
                        upcoming_df['win_probability'] = probabilities

                        # Confirm race details
                        st.write(f"### Predictions for: {session.event['EventName']} on {session.date.date()}")

                        st.write("### Win Probabilities")
                        st.dataframe(upcoming_df[['driver', 'win_probability']].sort_values(by='win_probability', ascending=False))

            except Exception as e:
                st.error(f"Error predicting selected race: {e}")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
