import os
import fastf1
import pandas as pd
import streamlit as st
from datetime import timezone
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
    df['target'] = (df['finish_position'] == 1).astype(int)  # Predict_
