import os
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

# Ensure FastF1 cache directory exists
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)

# Initialise FastF1 cache
fastf1.Cache.enable_cache(cache_dir)

# Streamlit App
st.title("F1 Race Win Probability Predictor")
st.write("Click the button below to generate win probabilities for the next race.")

if st.button('Predict Next Race Probabilities'):

        # Step 1: Data collection
    years = list(range(2018, datetime.now().year + 1))
    all_race_data = []
    
    st.write("Collecting race data...")
    
    # Add a Streamlit progress bar
    progress_text = "Loading race data..."
    progress_bar = st.progress(0, text=progress_text)
    total_races = sum(len(fastf1.get_event_schedule(year)) for year in years)
    race_counter = 0
    
    for year in years:
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
                race_counter += 1
                progress_bar.progress(min(race_counter / total_races, 1.0), text=f"{progress_text} ({race_counter}/{total_races})")
                continue
    
            results = session.results
            if results is None:
                race_counter += 1
                progress_bar.progress(min(race_counter / total_races, 1.0), text=f"{progress_text} ({race_counter}/{total_races})")
                continue
    
            # Calculate the number of laps completed by each driver
            if not session.laps.empty:
                laps_completed = session.laps.groupby('Driver')['LapNumber'].max().reset_index()
                laps_completed.rename(columns={'LapNumber': 'laps'}, inplace=True)
            else:
                # If laps data is missing, create an empty DataFrame with correct structure
                laps_completed = pd.DataFrame({'Driver': results['Abbreviation'], 'laps': np.nan})
    
            # Merge the results with the laps completed data
            results = results.merge(laps_completed, left_on='Abbreviation', right_on='Driver', how='left')
            results.drop(columns=['Driver'], inplace=True)  # Drop the redundant 'Driver' column
    
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
    
            race_counter += 1
            progress_bar.progress(min(race_counter / total_races, 1.0), text=f"{progress_text} ({race_counter}/{total_races})")

progress_bar.empty()  # Remove the progress bar when done


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

        # Add race number per driver to order races
        df.sort_values(by=['driver', 'year', 'round'], inplace=True)
        df['race_number'] = df.groupby('driver').cumcount() + 1

        # Create average position in last 3 and last 5 races
        df['avg_position_last_3'] = df.groupby('driver')['position'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        df['avg_position_last_5'] = df.groupby('driver')['position'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        st.write("Added average positions for last 3 and 5 races.")

        features = ['grid_position', 'team_encoded', 'points', 'laps', 'fastest_lap_speed', 'avg_position_last_3', 'avg_position_last_5']
        df = df.dropna(subset=features)

        X = df[features]
        y = df['win']

        # Step 3: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 4: Model training with calibration (no sample weight needed now)
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
                    # Prepare history data for new race predictions
                    history = df.copy()

                    for index, row in results.iterrows():
                        driver = row['Abbreviation']

                        # Get last 3 and 5 races for the driver
                        driver_history = history[history['driver'] == driver].tail(5)

                        avg_pos_last_3 = driver_history.tail(3)['position'].mean() if not driver_history.tail(3).empty else np.nan
                        avg_pos_last_5 = driver_history['position'].mean() if not driver_history.empty else np.nan

                        race_data.append({
                            'grid_position': row['GridPosition'],
                            'team_encoded': le_team.transform([row['TeamName']])[0] if row['TeamName'] in le_team.classes_ else 0,
                            'points': row['Points'],
                            'laps': row['Laps'],
                            'fastest_lap_speed': row['FastestLapSpeed'],
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

            except Exception as e:
                st.error(f"Failed to load next race session: {e}")
