import os
import fastf1
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Set up cache
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

st.title("F1 FastF1 Test App: Race Results & Next Race Prediction (with XGBoost)")

# --- Dropdown to select year and round ---
year_options = list(range(2018, 2025 + 1))
round_options = list(range(1, 24 + 1))

year = st.selectbox("Select Year", options=year_options, index=year_options.index(2023))
round_number = st.selectbox("Select Round", options=round_options, index=0)

# --- Dropdown for next race location ---
next_race_location = st.text_input("Next Race Location (Manual Input)", value="Suzuka")

# --- Load race data button ---
if st.button("Load Race Data"):
    try:
        st.write(f"Loading race data for **Year {year}, Round {round_number}**...")

        session = fastf1.get_session(year, round_number, 'R')
        session.load()

        st.success(f"✅ Loaded: {session.event['EventName']} ({session.date.date()})")

        # Show session info
        st.write("### Session Info:")
        st.json(session.event.to_dict())

        # Prepare race results DataFrame
        results = session.results
        if results is not None and not results.empty:
            df_results = results[['Abbreviation', 'Position', 'TeamName', 'Points']].sort_values(by='Position')
            df_results.columns = ['Driver', 'Finish Position', 'Team', 'Points']
            st.write("### Final Race Results")
            st.dataframe(df_results)
        else:
            st.warning("No results available for this session.")

        # Load historical data for model training
        st.write("Loading historical data...")
        data = []
        for hist_year in range(2018, 2025):
            for rnd in range(1, 25):
                try:
                    hist_session = fastf1.get_session(hist_year, rnd, 'R')
                    hist_session.load()
                    hist_results = hist_session.results
                    if hist_results is not None and not hist_results.empty:
                        for _, row in hist_results.iterrows():
                            data.append({
                                'Year': hist_year,
                                'Round': rnd,
                                'Location': hist_session.event['EventName'],
                                'Driver': row.Abbreviation,
                                'Team': row.TeamName,
                                'Grid Position': row.GridPosition,
                                'Finish Position': row.Position,
                                'Points': row.Points
                            })
                except Exception:
                    continue

        df_hist = pd.DataFrame(data)
        st.write("Historical data collected:", df_hist.shape)

        # Encode categorical variables
        le_driver = LabelEncoder()
        le_team = LabelEncoder()
        le_location = LabelEncoder()

        df_hist['Driver_encoded'] = le_driver.fit_transform(df_hist['Driver'])
        df_hist['Team_encoded'] = le_team.fit_transform(df_hist['Team'])
        df_hist['Location_encoded'] = le_location.fit_transform(df_hist['Location'])

        # Define features and target
        features = ['Driver_encoded', 'Team_encoded', 'Grid Position', 'Location_encoded']
        df_hist['Winner'] = (df_hist['Finish Position'] == 1).astype(int)

        X = df_hist[features]
        y = df_hist['Winner']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train XGBoost model
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        calibrated_model = CalibratedClassifierCV(model, cv=3)
        calibrated_model.fit(X_train, y_train)

        st.success("✅ Model trained successfully!")

        # Prepare input data for next race prediction
        input_data = df_results.copy()
        input_data['Driver_encoded'] = le_driver.transform(input_data['Driver'])
        input_data['Team_encoded'] = le_team.transform(input_data['Team'])
        input_data['Grid Position'] = 5  # Assume average start position
        input_data['Location_encoded'] = le_location.transform(
            [next_race_location] * len(input_data)
        )

        X_input = input_data[features]
        probs = calibrated_model.predict_proba(X_input)[:, 1]  # Probability of winning

        input_data['Win Probability (%)'] = (probs * 100).round(2)

        # Display predictions
        st.write("### Win Probability Predictions for Next Race")
        st.dataframe(input_data[['Driver', 'Team', 'Win Probability (%)']].sort_values(by='Win Probability (%)', ascending=False))

    except Exception as e:
        st.error(f"❌ Error: {e}")
