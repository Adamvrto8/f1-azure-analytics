# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook 01 — Setup & F1 Data Ingestion (v2)
# MAGIC 
# MAGIC Uses the **Jolpica F1 API** (successor to Ergast, which shut down in early 2025).
# MAGIC 
# MAGIC This notebook:
# MAGIC 1. Connects Databricks to Azure Data Lake Storage Gen2
# MAGIC 2. Pulls F1 data from the Jolpica API
# MAGIC 3. Saves raw JSON into the **bronze** container

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configure Storage Connection
# MAGIC 
# MAGIC **IMPORTANT:** Paste your storage account access key below.
# MAGIC 
# MAGIC To get it: Azure Portal → f1analyticslake2 → Security + networking → Access keys → Show → Copy key1

# COMMAND ----------

storage_account_name = "f1analyticslake2"

# >>> PASTE YOUR KEY BETWEEN THE QUOTES BELOW <<<
storage_account_key = "PASTE_YOUR_KEY_HERE"

# Set Spark config for ADLS Gen2 access
spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net",
    storage_account_key
)

# Define base paths
bronze_path = f"abfss://bronze@{storage_account_name}.dfs.core.windows.net"
silver_path = f"abfss://silver@{storage_account_name}.dfs.core.windows.net"
gold_path   = f"abfss://gold@{storage_account_name}.dfs.core.windows.net"

print(f"✓ Storage configured: {storage_account_name}")
print(f"  Bronze: {bronze_path}")
print(f"  Silver: {silver_path}")
print(f"  Gold:   {gold_path}")

# COMMAND ----------

# Test the connection
try:
    dbutils.fs.ls(bronze_path)
    print("✓ Successfully connected to bronze container!")
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("  → Check that your storage key is correct.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Ingest F1 Data from Jolpica API
# MAGIC 
# MAGIC The [Jolpica F1 API](https://api.jolpi.ca/ergast/f1/) is the community successor to Ergast.
# MAGIC Same data format, just a different URL.

# COMMAND ----------

import requests
import json
import time

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
SEASONS = range(2018, 2026)  # 2018 through 2025

def fetch_f1(endpoint, limit=1000, max_retries=3):
    """Fetch data from Jolpica F1 API with retries."""
    url = f"{JOLPICA_BASE}/{endpoint}.json?limit={limit}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited — wait and retry
                wait_time = 10 * (attempt + 1)
                print(f"  ⏳ Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ Status {response.status_code}: {url}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
    
    return None

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2a: Race Schedule & Circuit Info

# COMMAND ----------

all_races = []

for season in SEASONS:
    print(f"Fetching {season} schedule...")
    data = fetch_f1(f"{season}")
    if data:
        races = data["MRData"]["RaceTable"]["Races"]
        all_races.extend(races)
        print(f"  ✓ {len(races)} races")
    time.sleep(1)  # Respect rate limits (200 req/hr)

print(f"\nTotal: {len(all_races)} race schedules fetched")

races_json = json.dumps(all_races, indent=2)
dbutils.fs.put(f"{bronze_path}/races/races_raw.json", races_json, overwrite=True)
print(f"✓ Saved to {bronze_path}/races/")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2b: Race Results

# COMMAND ----------

all_results = []

for season in SEASONS:
    print(f"Fetching {season} race results...")
    data = fetch_f1(f"{season}/results", limit=1000)
    if data:
        races = data["MRData"]["RaceTable"]["Races"]
        all_results.extend(races)
        print(f"  ✓ {len(races)} races with results")
    time.sleep(1)

print(f"\nTotal: {len(all_results)} race results fetched")

results_json = json.dumps(all_results, indent=2)
dbutils.fs.put(f"{bronze_path}/race_results/race_results_raw.json", results_json, overwrite=True)
print(f"✓ Saved to {bronze_path}/race_results/")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2c: Qualifying Results

# COMMAND ----------

all_qualifying = []

for season in SEASONS:
    print(f"Fetching {season} qualifying...")
    data = fetch_f1(f"{season}/qualifying", limit=1000)
    if data:
        races = data["MRData"]["RaceTable"]["Races"]
        all_qualifying.extend(races)
        print(f"  ✓ {len(races)} qualifying sessions")
    time.sleep(1)

print(f"\nTotal: {len(all_qualifying)} qualifying sessions fetched")

qualifying_json = json.dumps(all_qualifying, indent=2)
dbutils.fs.put(f"{bronze_path}/qualifying/qualifying_raw.json", qualifying_json, overwrite=True)
print(f"✓ Saved to {bronze_path}/qualifying/")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2d: Pit Stops
# MAGIC 
# MAGIC Pit stops require per-race requests. This takes a bit longer.

# COMMAND ----------

all_pitstops = []
errors = 0

for season in SEASONS:
    # Get race count for this season
    schedule = fetch_f1(f"{season}")
    if not schedule:
        print(f"  ✗ Could not get {season} schedule, skipping...")
        continue
    
    races = schedule["MRData"]["RaceTable"]["Races"]
    season_pitstops = 0
    
    for race in races:
        round_num = race["round"]
        data = fetch_f1(f"{season}/{round_num}/pitstops", limit=1000)
        if data:
            race_data = data["MRData"]["RaceTable"]["Races"]
            if race_data and "PitStops" in race_data[0]:
                pitstop_record = {
                    "season": str(season),
                    "round": round_num,
                    "raceName": race.get("raceName", ""),
                    "PitStops": race_data[0]["PitStops"]
                }
                all_pitstops.append(pitstop_record)
                season_pitstops += len(race_data[0]["PitStops"])
        else:
            errors += 1
        time.sleep(0.5)
    
    print(f"  ✓ {season}: {season_pitstops} pit stops from {len(races)} races")

print(f"\nTotal: {len(all_pitstops)} race records with pit stop data")
if errors > 0:
    print(f"  ({errors} failed requests — partial data is fine for now)")

pitstops_json = json.dumps(all_pitstops, indent=2)
dbutils.fs.put(f"{bronze_path}/pitstops/pitstops_raw.json", pitstops_json, overwrite=True)
print(f"✓ Saved to {bronze_path}/pitstops/")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2e: Lap Times
# MAGIC 
# MAGIC ⚠️ This is the slowest section — fetches lap-by-lap data per race.
# MAGIC For 8 seasons this can take 15-30 minutes. Be patient!

# COMMAND ----------

all_laptimes = []

for season in SEASONS:
    schedule = fetch_f1(f"{season}")
    if not schedule:
        continue
    
    races = schedule["MRData"]["RaceTable"]["Races"]
    season_count = 0
    
    for race in races:
        round_num = race["round"]
        race_name = race.get("raceName", "")
        
        # Fetch all laps for this race in one call (high limit)
        data = fetch_f1(f"{season}/{round_num}/laps", limit=2000)
        if data:
            race_data = data["MRData"]["RaceTable"]["Races"]
            if race_data and "Laps" in race_data[0]:
                for lap_data in race_data[0]["Laps"]:
                    for timing in lap_data.get("Timings", []):
                        all_laptimes.append({
                            "season": str(season),
                            "round": round_num,
                            "raceName": race_name,
                            "lap": lap_data["number"],
                            "driverId": timing["driverId"],
                            "position": timing.get("position", ""),
                            "time": timing.get("time", "")
                        })
                        season_count += 1
        time.sleep(0.5)
    
    print(f"  ✓ {season}: {season_count:,} lap time records")

print(f"\nTotal: {len(all_laptimes):,} lap time records fetched")

laptimes_json = json.dumps(all_laptimes, indent=2)
dbutils.fs.put(f"{bronze_path}/laptimes/laptimes_raw.json", laptimes_json, overwrite=True)
print(f"✓ Saved to {bronze_path}/laptimes/")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2f: Driver & Constructor Standings

# COMMAND ----------

all_driver_standings = []
all_constructor_standings = []

for season in SEASONS:
    # Driver standings
    data = fetch_f1(f"{season}/driverStandings")
    if data:
        standings = data["MRData"]["StandingsTable"]["StandingsLists"]
        if standings:
            all_driver_standings.append({
                "season": str(season),
                "DriverStandings": standings[0]["DriverStandings"]
            })
    time.sleep(1)
    
    # Constructor standings
    data = fetch_f1(f"{season}/constructorStandings")
    if data:
        standings = data["MRData"]["StandingsTable"]["StandingsLists"]
        if standings:
            all_constructor_standings.append({
                "season": str(season),
                "ConstructorStandings": standings[0]["ConstructorStandings"]
            })
    time.sleep(1)
    
    print(f"  ✓ {season} standings fetched")

# Save driver standings
driver_json = json.dumps(all_driver_standings, indent=2)
dbutils.fs.put(f"{bronze_path}/standings/driver_standings_raw.json", driver_json, overwrite=True)

# Save constructor standings
constructor_json = json.dumps(all_constructor_standings, indent=2)
dbutils.fs.put(f"{bronze_path}/standings/constructor_standings_raw.json", constructor_json, overwrite=True)

print(f"\n✓ Driver standings: {len(all_driver_standings)} seasons")
print(f"✓ Constructor standings: {len(all_constructor_standings)} seasons")
print(f"Saved to {bronze_path}/standings/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Verify Bronze Layer

# COMMAND ----------

print("=" * 50)
print("  BRONZE LAYER CONTENTS")
print("=" * 50)

folders = ["races", "race_results", "qualifying", "pitstops", "laptimes", "standings"]
total_size = 0

for folder in folders:
    try:
        files = dbutils.fs.ls(f"{bronze_path}/{folder}/")
        folder_size = sum(f.size for f in files)
        total_size += folder_size
        print(f"\n📁 {folder}/")
        for f in files:
            size_kb = f.size / 1024
            print(f"   └── {f.name} ({size_kb:.1f} KB)")
    except Exception:
        print(f"\n📁 {folder}/ — not found")

print(f"\n{'=' * 50}")
print(f"  Total bronze data: {total_size / 1024:.1f} KB")
print(f"{'=' * 50}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Quick Data Preview

# COMMAND ----------

# Read race results as a Spark DataFrame
results_df = (
    spark.read
    .option("multiline", "true")
    .json(f"{bronze_path}/race_results/race_results_raw.json")
)

print("Race results columns:")
print(results_df.columns)
print(f"\nTotal race records: {results_df.count()}")

# COMMAND ----------

# Preview the data
display(results_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Notebook Complete!
# MAGIC 
# MAGIC **What we accomplished:**
# MAGIC - Connected Databricks to ADLS Gen2 storage
# MAGIC - Ingested race schedules, results, qualifying, pit stops, lap times, and standings (2018–2025)
# MAGIC - Saved everything as raw JSON in the bronze container
# MAGIC - Verified data landed correctly
# MAGIC 
# MAGIC **Data source:** Jolpica F1 API (https://api.jolpi.ca/ergast/f1/)
# MAGIC 
# MAGIC **Next:** `02_bronze_to_delta` — Convert raw JSON to Delta tables
