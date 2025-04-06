import os
import fastf1
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

# Set up cache
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Streamlit app
st.title("F1 FastF1 Test App: Explore Race Results")

st.write("Use the dropdown menus below to select a season and round number to display race results.")

# Manual year and round input as dropdown menus
year_options = list(range(2018, 2025 + 1))
round_options = list(range(1, 24 + 1))

year = st.selectbox("Select Year", options=year_options, index=year_options.index(2023))
round_number = st.selectbox("Select Round", options=round_options, index=0)

# Declare session globally
session = None

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

# --- NEW: Machine learning model button ---
if st.button("Generate Win Predictions"):
    try:
        st.write("Preparing data for prediction...")

        # For simplicity, use the current session only for training (demo purpose)
        session = fastf1.get_session(year, round_number, 'R')
        session.load()

        results = session.results

        if results is None or results.empty:
            st.warning("No results data available to generate predictions.")
        else:
            df = results.copy()

            # Simple feature engineering
            df['win'] = (df['Position'] == 1).astype(int)

            le_driver = LabelEncoder()
            le_team = LabelEncoder()

            df['driver_encoded'] = le_driver.fit_transform(df['Abbreviation'])
            df['team_encoded'] = le_team.fit_transform(df['TeamName'])

            # Basic features
            df['grid_position'] = df['GridPosition']
            df['points'] = df['Points']

            # Handle laps
            if not session.laps.empty:
                laps_completed = session.laps.groupby('Driver')['LapNumber'].max().reset_index()
                laps_completed.rename(columns={'LapNumber': 'laps'}, inplace=True)
                df = df.merge(laps_completed, left_on='Abbreviation', right_on='Driver', how='left')
                df['laps'] = df['laps'].fillna(0)
                df.drop(columns=['Driver'], inplace=True)
            else:
                df['laps'] = 0

            # Features and target
            features = ['grid_position', 'team_encoded', 'points', 'laps']
            X = df[features]
            y = df['win']

            # Simple train/test split (we will use same data to train and predict for demo purposes)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model training
            base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
            calibrated_model.fit(X_train, y_train)

            # Predict probabilities
            probabilities = calibrated_model.predict_proba(X)[:, 1]
            df['win_probability'] = probabilities

            st.success(f"Predictions generated for: {session.event['EventName']} on {session.date.date()}")

            # Display predictions
            df_predictions = df[['Abbreviation', 'win_probability']].sort_values(by='win_probability', ascending=False)
            df_predictions.columns = ['Driver', 'Win Probability']
            st.write("### Win Probabilities")
            st.dataframe(df_predictions)

    except Exception as e:
        st.error(f"❌ Error generating predictions: {e}")
