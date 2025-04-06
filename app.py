import os
import fastf1
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Set up cache for FastF1
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Paths for persistent files
data_file = 'historical_data.csv'
model_file = 'f1_model.pkl'

st.title("F1 FastF1 App: Race Results & Persistent ML Predictions")

# --- Load or fetch historical data ---
@st.cache_data(show_spinner=True)
def load_or_fetch_historical_data():
    if os.path.exists(data_file):
        st.info("Loading historical data from disk...")
        return pd.read_csv(data_file)
    else:
        st.info("Downloading full historical data (may take a few minutes)...")
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
        df = pd.DataFrame(data)
        df.to_csv(data_file, index=False)
        st.success(f"✅ Historical data downloaded and saved to {data_file}")
        return df

with st.spinner("Loading historical race data..."):
    df_hist = load_or_fetch_historical_data()
    st.success(f"✅ Historical data ready: {df_hist.shape[0]} records.")

# --- Encode data ---
le_driver = LabelEncoder()
le_team = LabelEncoder()
le_location = LabelEncoder()

df_hist['Driver_encoded'] = le_driver.fit_transform(df_hist['Driver'])
df_hist['Team_encoded'] = le_team.fit_transform(df_hist['Team'])
df_hist['Location_encoded'] = le_location.fit_transform(df_hist['Location'])
df_hist['Winner'] = (df_hist['Finish Position'] == 1).astype(int)

features = ['Driver_encoded', 'Team_encoded', 'Grid Position', 'Location_encoded']
X = df_hist[features]
y = df_hist['Winner']

# --- Load or train model ---
if os.path.exists(model_file):
    st.info("Loading trained model from disk...")
    calibrated_model = joblib.load(model_file)
    st.success("✅ Model loaded successfully!")
else:
    st.info("Training new model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    calibrated_model = CalibratedClassifierCV(model, cv=3)
    calibrated_model.fit(X_train, y_train)
    joblib.dump(calibrated_model, model_file)
    st.success(f"✅ Model trained and saved to {model_file}")

# --- User selects last completed race ---
year_options = list(range(2018, 2025))
round_options = list(range(1, 24 + 1))

year = st.selectbox("Select Last Completed Race Year", options=year_options, index=year_options.index(2024))
round_number = st.selectbox("Select Last Completed Round", options=round_options, index=0)

available_locations = df_hist['Location'].unique()
st.write("Available Locations for Next Race Input:")
st.write(sorted(available_locations))

next_race_location = st.text_input("Next Race Location (manual input, must match list above)", value=available_locations[0])

if next_race_location not in available_locations:
    st.warning(f"⚠️ Location '{next_race_location}' not found in historical data. Please choose from the list above.")
else:
    if st.button("Load Last Race & Predict Next Race"):
        try:
            st.write(f"Loading last race data for Year {year}, Round {round_number}...")
            session = fastf1.get_session(year, round_number, 'R')
            session.load()

            st.success(f"✅ Loaded: {session.event['EventName']} ({session.date.date()})")

            # Prepare driver/team input
            results = session.results
            if results is not None and not results.empty:
                df_results = results[['Abbreviation', 'TeamName']].copy()
                df_results.columns = ['Driver', 'Team']
                st.write("### Last Race Drivers & Teams")
                st.dataframe(df_results)

                # Prepare input data for prediction
                input_data = df_results.copy()
                input_data['Driver_encoded'] = le_driver.transform(input_data['Driver'])
                input_data['Team_encoded'] = le_team.transform(input_data['Team'])
                input_data['Grid Position'] = 5  # Assume average position
                input_data['Location_encoded'] = le_location.transform([next_race_location] * len(input_data))

                X_input = input_data[features]
                probs = calibrated_model.predict_proba(X_input)[:, 1]  # Probability of winning

                input_data['Win Probability (%)'] = (probs * 100).round(2)

                st.write("### Win Probability Predictions for Next Race")
                st.dataframe(input_data[['Driver', 'Team', 'Win Probability (%)']].sort_values(by='Win Probability (%)', ascending=False))

            else:
                st.warning("No results available for this session.")

        except Exception as e:
            st.error(f"❌ Error loading session: {e}")
