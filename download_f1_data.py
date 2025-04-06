import os
import fastf1
import pandas as pd
import streamlit as st

# Set up FastF1 cache
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

st.title("F1 Historical Data Downloader")

# Output file name
output_file = 'historical_data.csv'

# Prepare storage
data = []

with st.spinner("Downloading F1 historical data..."):
    for year in range(2018, 2025):
        for rnd in range(1, 25):
            try:
                session = fastf1.get_session(year, rnd, 'R')
                session.load()
                results = session.results

                if results is not None and not results.empty:
                    for _, row in results.iterrows():
                        data.append({
                            'Year': year,
                            'Round': rnd,
                            'Location': session.event['EventName'],
                            'Driver': row.Abbreviation,
                            'Team': row.TeamName,
                            'Grid Position': row.GridPosition,
                            'Finish Position': row.Position,
                            'Points': row.Points
                        })
                    st.write(f"✅ {year} Round {rnd} - {session.event['EventName']}")
                else:
                    st.write(f"⚠️ No results for {year} Round {rnd}")

            except Exception as e:
                st.write(f"❌ Skipped {year} Round {rnd}: {e}")
                continue

# Convert to DataFrame
df = pd.DataFrame(data)

if not df.empty:
    # Save to file
    df.to_csv(output_file, index=False)

    st.success(f"✅ Download complete: {df.shape[0]} rows.")
    st.write("### Preview of your data:")
    st.dataframe(df.head())

    # Add download button
    st.download_button(
        label="Download historical_data.csv",
        data=df.to_csv(index=False),
        file_name=output_file,
        mime='text/csv'
    )
else:
    st.error("No data downloaded. Please check FastF1 availability.")
