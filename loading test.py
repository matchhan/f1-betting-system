import os
import fastf1
import pandas as pd
import streamlit as st

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

if st.button("Load Race Data"):
    try:
        st.write(f"Loading race data for **Year {year}, Round {round_number}**...")

        session = fastf1.get_session(year, round_number, 'R')
        session.load()

        st.success(f"✅ Successfully loaded: {session.event['EventName']} ({session.date.date()})")

        # Show session info (Fixed!)
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
