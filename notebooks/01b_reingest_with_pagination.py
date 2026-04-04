# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook 01b — Re-ingest F1 Data (with pagination)
# MAGIC 
# MAGIC The Jolpica API has a **max limit of 100 results per request**.
# MAGIC This notebook fetches data **per race** to get complete results,
# MAGIC and uses **offset pagination** for large datasets like lap times.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Storage Connection

# COMMAND ----------

storage_account_name = "f1analyticslake2"

# >>> PASTE YOUR KEY BELOW <<<
storage_account_key = "PASTE_YOUR_KEY_HERE"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net",
    storage_account_key
)

bronze_path = f"abfss://bronze@{storage_account_name}.dfs.core.windows.net"
print("✓ Storage connected")

# COMMAND ----------

import requests
import json
import time

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
SEASONS = range(2018, 2026)

def fetch_f1(endpoint, limit=100, offset=0, max_retries=3):
    """Fetch from Jolpica API. Max limit is 100."""
    url = f"{JOLPICA_BASE}/{endpoint}.json?limit={limit}&offset={offset}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  ⏳ Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                if attempt < max_retries - 1:
                    time.sleep(3)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(3)
    return None

def fetch_all_pages(endpoint, limit=100):
    """Fetch all pages for an endpoint using offset pagination."""
    all_items = []
    offset = 0
    
    while True:
        data = fetch_f1(endpoint, limit=limit, offset=offset)
        if not data:
            break
        
        total = int(data["MRData"].get("total", 0))
        races = data["MRData"]["RaceTable"]["Races"]
        all_items.extend(races)
        
        offset += limit
        if offset >= total:
            break
        time.sleep(0.5)
    
    return all_items

# COMMAND ----------

# MAGIC %md
# MAGIC ## Race Results (per race — complete data)

# COMMAND ----------

all_results = []

for season in SEASONS:
    # First get the schedule to know how many races
    schedule = fetch_f1(f"{season}")
    if not schedule:
        continue
    races = schedule["MRData"]["RaceTable"]["Races"]
    
    season_count = 0
    for race in races:
        round_num = race["round"]
        data = fetch_f1(f"{season}/{round_num}/results", limit=100)
        if data:
            race_results = data["MRData"]["RaceTable"]["Races"]
            if race_results:
                all_results.extend(race_results)
                result_count = len(race_results[0].get("Results", []))
                season_count += result_count
        time.sleep(0.5)
    
    print(f"  ✓ {season}: {len(races)} races, {season_count} driver results")

print(f"\nTotal: {len(all_results)} race records")

results_json = json.dumps(all_results, indent=2)
dbutils.fs.put(f"{bronze_path}/race_results/race_results_raw.json", results_json, overwrite=True)
print(f"✓ Saved to bronze/race_results/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Qualifying (per race)

# COMMAND ----------

all_qualifying = []

for season in SEASONS:
    schedule = fetch_f1(f"{season}")
    if not schedule:
        continue
    races = schedule["MRData"]["RaceTable"]["Races"]
    
    season_count = 0
    for race in races:
        round_num = race["round"]
        data = fetch_f1(f"{season}/{round_num}/qualifying", limit=100)
        if data:
            race_qual = data["MRData"]["RaceTable"]["Races"]
            if race_qual:
                all_qualifying.extend(race_qual)
                qual_count = len(race_qual[0].get("QualifyingResults", []))
                season_count += qual_count
        time.sleep(0.5)
    
    print(f"  ✓ {season}: {len(races)} races, {season_count} qualifying results")

print(f"\nTotal: {len(all_qualifying)} qualifying race records")

qualifying_json = json.dumps(all_qualifying, indent=2)
dbutils.fs.put(f"{bronze_path}/qualifying/qualifying_raw.json", qualifying_json, overwrite=True)
print(f"✓ Saved to bronze/qualifying/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pit Stops (per race)

# COMMAND ----------

all_pitstops = []

for season in SEASONS:
    schedule = fetch_f1(f"{season}")
    if not schedule:
        continue
    races = schedule["MRData"]["RaceTable"]["Races"]
    
    season_stops = 0
    for race in races:
        round_num = race["round"]
        data = fetch_f1(f"{season}/{round_num}/pitstops", limit=100)
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
                season_stops += len(race_data[0]["PitStops"])
        time.sleep(0.5)
    
    print(f"  ✓ {season}: {season_stops} pit stops")

print(f"\nTotal: {len(all_pitstops)} race records with pit stops")

pitstops_json = json.dumps(all_pitstops, indent=2)
dbutils.fs.put(f"{bronze_path}/pitstops/pitstops_raw.json", pitstops_json, overwrite=True)
print(f"✓ Saved to bronze/pitstops/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lap Times (per race with pagination)
# MAGIC 
# MAGIC ⚠️ This is the slowest section — each race can have 1000+ lap records.
# MAGIC Uses offset pagination. Expect 20-40 minutes.

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
        offset = 0
        
        while True:
            data = fetch_f1(f"{season}/{round_num}/laps", limit=100, offset=offset)
            if not data:
                break
            
            total = int(data["MRData"].get("total", 0))
            race_data = data["MRData"]["RaceTable"]["Races"]
            
            if not race_data or "Laps" not in race_data[0]:
                break
            
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
            
            offset += 100
            if offset >= total:
                break
            time.sleep(0.3)
        
        time.sleep(0.3)
    
    print(f"  ✓ {season}: {season_count:,} lap times")

print(f"\nTotal: {len(all_laptimes):,} lap time records")

laptimes_json = json.dumps(all_laptimes, indent=2)
dbutils.fs.put(f"{bronze_path}/laptimes/laptimes_raw.json", laptimes_json, overwrite=True)
print(f"✓ Saved to bronze/laptimes/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Standings (paginated)

# COMMAND ----------

all_driver_standings = []
all_constructor_standings = []

for season in SEASONS:
    data = fetch_f1(f"{season}/driverStandings", limit=100)
    if data:
        standings = data["MRData"]["StandingsTable"]["StandingsLists"]
        if standings:
            all_driver_standings.append({
                "season": str(season),
                "DriverStandings": standings[0]["DriverStandings"]
            })
    time.sleep(1)
    
    data = fetch_f1(f"{season}/constructorStandings", limit=100)
    if data:
        standings = data["MRData"]["StandingsTable"]["StandingsLists"]
        if standings:
            all_constructor_standings.append({
                "season": str(season),
                "ConstructorStandings": standings[0]["ConstructorStandings"]
            })
    time.sleep(1)
    
    print(f"  ✓ {season} standings")

dbutils.fs.put(f"{bronze_path}/standings/driver_standings_raw.json", 
               json.dumps(all_driver_standings, indent=2), overwrite=True)
dbutils.fs.put(f"{bronze_path}/standings/constructor_standings_raw.json", 
               json.dumps(all_constructor_standings, indent=2), overwrite=True)

print(f"\n✓ Driver standings: {len(all_driver_standings)} seasons")
print(f"✓ Constructor standings: {len(all_constructor_standings)} seasons")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Complete Data

# COMMAND ----------

print("=" * 60)
print("  COMPLETE BRONZE DATA SUMMARY")
print("=" * 60)

folders = ["race_results", "qualifying", "pitstops", "laptimes", "standings"]
for folder in folders:
    try:
        files = dbutils.fs.ls(f"{bronze_path}/{folder}/")
        total_kb = sum(f.size for f in files) / 1024
        print(f"\n📁 {folder}/")
        for f in files:
            print(f"   └── {f.name} ({f.size/1024:.1f} KB)")
    except:
        print(f"\n📁 {folder}/ — not found")

print(f"\n{'=' * 60}")
print("Now re-run notebook 02 to rebuild Delta tables with complete data!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Done!
# MAGIC 
# MAGIC Now go back and **re-run notebook 02** (bronze_to_delta) to rebuild 
# MAGIC your Delta tables with the complete dataset. You should see much 
# MAGIC higher row counts this time.
# MAGIC 
# MAGIC **Expected counts:**
# MAGIC - Race results: ~3,500+ rows (20 drivers × ~170 races)
# MAGIC - Qualifying: ~3,500+ rows
# MAGIC - Pit stops: ~7,000+ rows
# MAGIC - Lap times: ~170,000+ rows
# MAGIC - Standings: ~170 driver, ~80 constructor
