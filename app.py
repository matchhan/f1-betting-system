import os
import fastf1
import pandas as pd
import streamlit as st

# Set up FastF1 cache
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Path for saving data
data_file = 'historical_data.csv'

st.title("F1 Data Builder: Historical Dataset Download and Save")

# Step 1: Load existing data if available
if os.path.exists(data_file):
    st.info("Loading existing historical data from disk...")
    df_hist = pd.read_csv(data_file)
    st.write(f"✅ Existing data found: {df_hist.shape[0]} records.")
else:
    df_hist = pd.DataFrame()
    st.info("No existing data found. Starting fresh download...")

# Step 2: Build list of already collected sessions
if not df_hist.empty:
    existing_sessions = set(zip(df_hist['Year'], df_hist['Round']))
else:
    existing_sessions = set()

# Step 3: Download missing data
new_data = []

with st.spinner("Downloading missing race data..."):
    total_downloads = 0
    for year in range(2018, 2025):
        for rnd in range(1, 25):
            if (year, rnd) in existing_sessions:
                continue  # Skip already collected
            try:
                session = fastf1.get_session(year, rnd, 'R')
                session.load()
                results = session.results
                if results is not None and not results.empty:
                    for _, row in results.iterrows():
                        new_data.append({
                            'Year': year,
                            'Round': rnd,
                            'Location': session.event['EventName'],
                            'Driver': row.Abbreviation,
                            'Team': row.TeamName,
                            'Grid Position': row.GridPosition,
                            'Finish Position': row.Position,
                            'Points': row.Points
                        })
                    total_downloads += 1
                    st.write(f"✅ Downloaded: {year} Round {rnd} - {session.event['EventName']}")
            except Exception as e:
                st.write(f"⚠️ Skipped {year} Round {rnd}: {e}")
                continue

# Step 4: Save data
if new_data:
    df_new = pd.DataFrame(new_data)
    df_combined = pd.concat([df_hist, df_new], ignore_index=True)
    df_combined.to_csv(data_file, index=False)
    st.success(f"✅ New data saved! Total records: {df_combined.shape[0]}")
else:
    df_combined = df_hist
    st.info("No new data to add. Dataset is already up to date.")

# Step 5: Show preview
st.write("### Preview of saved dataset:")
st.dataframe(df_combined.head())
