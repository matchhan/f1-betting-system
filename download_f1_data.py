import os
import fastf1
import pandas as pd

# Set up FastF1 cache
cache_dir = './fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Output CSV file
output_file = 'historical_data.csv'

# Prepare storage
data = []

print("🚀 Starting F1 historical data download...")

# Loop through years and rounds
for year in range(2018, 2025):
    for rnd in range(1, 25):
        try:
            print(f"⏳ Fetching data for {year} Round {rnd}...")
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
                print(f"✅ Completed: {year} Round {rnd} - {session.event['EventName']}")
            else:
                print(f"⚠️ No results found for {year} Round {rnd}.")

        except Exception as e:
            print(f"❌ Skipped {year} Round {rnd}: {e}")
            continue

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv(output_file, index=False)

print(f"\n✅ All data saved to '{output_file}' ({df.shape[0]} rows).")
