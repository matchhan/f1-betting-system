import os
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

# Prepare FastF1 cache
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Streamlit App setup
st.title("F1 Win Probability Predictor")
st.write("Click the button below to load data, train the model, and predict win probabilities for the next race.")

if st.button("Generate Predictions"):

    try:
        # Step 1: Data collection
        st.write("Loading historical race data...")

        years = list(range(2018, datetime.now().year))
        all_race_data = []

        for year in years:
            try:
                schedule = fastf1.get_event_schedule(year)
            except Exception:
                continue  # Skip if schedule cannot be loaded

            for _, race in schedule.iterrows():
                round_number = race['RoundNumber']

                try:
                    session = fastf1.get_session(year, round_number, 'R')
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
                        'year': year,
                        'round': round_number,
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

            # Drop any rows with missing features
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

            # Step 4: Predict next race
            latest_year = datetime.now().year
            try:
                schedule = fastf1.get_event_schedule(latest_year)
                upcoming_races = schedule[schedule['Session1Date'] > datetime.now()]

                if upcoming_races.empty:
                    st.warning("No upcoming races found.")
                else:
                    next_race = upcoming_races.iloc[0]
                    round_number = next_race['RoundNumber']
                    race_name = next_race['EventName']
                    race_date = next_race['Session1Date'].date()

                    st.write(f"### Next Race: {race_name} on {race_date}")

                    session = fastf1.get_session(latest_year, round_number, 'R')
                    session.load()

                    if session.results is None:
                        st.warning("No results available yet for the next race (grid not set).")
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
                            st.warning("Not enough data to make predictions for the next race.")
                        else:
                            probabilities = calibrated_model.predict_proba(upcoming_df[features])[:, 1]
                            upcoming_df['win_probability'] = probabilities

                            st.write("### Win Probabilities for Next Race")
                            st.dataframe(upcoming_df[['driver', 'win_probability']].sort_values(by='win_probability', ascending=False))

            except Exception as e:
                st.error(f"Error predicting next race: {e}")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
