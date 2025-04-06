import os
import fastf1

# Create the cache directory programmatically (important for Streamlit Cloud!)
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)

# Enable cache
fastf1.Cache.enable_cache(cache_dir)

# Pick a known past race — safe data
year = 2023
round_number = 1  # Bahrain Grand Prix 2023

try:
    # Get the session
    session = fastf1.get_session(year, round_number, 'R')
    session.load()

    print(f"\n✅ Successfully loaded: {session.event['EventName']} ({session.date.date()})\n")

    # Print session info
    print("Session Info:")
    print(session.event)

    # Print results table
    results = session.results
    if results is not None and not results.empty:
        print("\nResults:")
        print(results[['Abbreviation', 'Position', 'TeamName', 'Points']])
    else:
        print("No results available for this session.")

    # Print lap data info
    if not session.laps.empty:
        print("\nLap data sample:")
        print(session.laps.head())
    else:
        print("No lap data available.")

except Exception as e:
    print(f"\n❌ Error loading session: {e}")
