# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook 02 — Bronze to Delta Tables
# MAGIC 
# MAGIC Converts raw JSON from the bronze container into structured **Delta tables**.
# MAGIC This gives us:
# MAGIC - Schema enforcement
# MAGIC - Fast SQL queries
# MAGIC - Time travel (versioning)
# MAGIC - Foundation for the silver layer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Storage Connection

# COMMAND ----------

storage_account_name = "f1analyticslake2"

# >>> PASTE YOUR KEY BETWEEN THE QUOTES BELOW <<<
storage_account_key = "PASTE_YOUR_KEY_HERE"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net",
    storage_account_key
)

bronze_path = f"abfss://bronze@{storage_account_name}.dfs.core.windows.net"
print("✓ Storage connected")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Bronze Database

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS f1_bronze")
spark.sql("USE f1_bronze")
print("✓ Database f1_bronze ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Race Results → Delta
# MAGIC 
# MAGIC The race results JSON is nested — each race contains an array of results.
# MAGIC We'll flatten it into one row per driver per race.

# COMMAND ----------

from pyspark.sql.functions import explode, col, lit
from pyspark.sql.types import *

# Read raw JSON
raw_results = (
    spark.read
    .option("multiline", "true")
    .json(f"{bronze_path}/race_results/race_results_raw.json")
)

print(f"Raw results records: {raw_results.count()}")
raw_results.printSchema()

# COMMAND ----------

# Flatten: one row per driver per race
results_flat = (
    raw_results
    .select(
        col("season"),
        col("round"),
        col("raceName"),
        col("Circuit.circuitId").alias("circuitId"),
        col("Circuit.circuitName").alias("circuitName"),
        col("Circuit.Location.country").alias("country"),
        col("date"),
        explode("Results").alias("result")
    )
    .select(
        "season", "round", "raceName", "circuitId", "circuitName", "country", "date",
        col("result.number").alias("driverNumber"),
        col("result.position").alias("position"),
        col("result.positionText").alias("positionText"),
        col("result.points").alias("points"),
        col("result.Driver.driverId").alias("driverId"),
        col("result.Driver.code").alias("driverCode"),
        col("result.Driver.givenName").alias("firstName"),
        col("result.Driver.familyName").alias("lastName"),
        col("result.Driver.nationality").alias("driverNationality"),
        col("result.Constructor.constructorId").alias("constructorId"),
        col("result.Constructor.name").alias("constructorName"),
        col("result.grid").alias("gridPosition"),
        col("result.laps").alias("lapsCompleted"),
        col("result.status").alias("status"),
        col("result.Time.millis").alias("timeMillis"),
        col("result.Time.time").alias("timeText"),
        col("result.FastestLap.rank").alias("fastestLapRank"),
        col("result.FastestLap.lap").alias("fastestLapNumber"),
        col("result.FastestLap.Time.time").alias("fastestLapTime"),
        col("result.FastestLap.AverageSpeed.speed").alias("fastestLapAvgSpeed"),
    )
)

# Cast numeric columns
results_final = (
    results_flat
    .withColumn("season", col("season").cast("int"))
    .withColumn("round", col("round").cast("int"))
    .withColumn("position", col("position").cast("int"))
    .withColumn("points", col("points").cast("float"))
    .withColumn("gridPosition", col("gridPosition").cast("int"))
    .withColumn("lapsCompleted", col("lapsCompleted").cast("int"))
    .withColumn("timeMillis", col("timeMillis").cast("long"))
    .withColumn("fastestLapRank", col("fastestLapRank").cast("int"))
    .withColumn("fastestLapNumber", col("fastestLapNumber").cast("int"))
    .withColumn("fastestLapAvgSpeed", col("fastestLapAvgSpeed").cast("float"))
)

print(f"Flattened results: {results_final.count()} rows")
display(results_final.limit(10))

# COMMAND ----------

# Save as Delta table
(
    results_final.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_bronze.race_results")
)

print("✓ f1_bronze.race_results saved as Delta table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Qualifying → Delta

# COMMAND ----------

raw_qualifying = (
    spark.read
    .option("multiline", "true")
    .json(f"{bronze_path}/qualifying/qualifying_raw.json")
)

qualifying_flat = (
    raw_qualifying
    .select(
        col("season"),
        col("round"),
        col("raceName"),
        col("Circuit.circuitId").alias("circuitId"),
        col("date"),
        explode("QualifyingResults").alias("qual")
    )
    .select(
        "season", "round", "raceName", "circuitId", "date",
        col("qual.number").alias("driverNumber"),
        col("qual.position").alias("position"),
        col("qual.Driver.driverId").alias("driverId"),
        col("qual.Driver.code").alias("driverCode"),
        col("qual.Driver.givenName").alias("firstName"),
        col("qual.Driver.familyName").alias("lastName"),
        col("qual.Constructor.constructorId").alias("constructorId"),
        col("qual.Constructor.name").alias("constructorName"),
        col("qual.Q1").alias("q1Time"),
        col("qual.Q2").alias("q2Time"),
        col("qual.Q3").alias("q3Time"),
    )
    .withColumn("season", col("season").cast("int"))
    .withColumn("round", col("round").cast("int"))
    .withColumn("position", col("position").cast("int"))
)

print(f"Qualifying records: {qualifying_flat.count()}")

(
    qualifying_flat.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_bronze.qualifying")
)

print("✓ f1_bronze.qualifying saved as Delta table")
display(qualifying_flat.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Pit Stops → Delta

# COMMAND ----------

raw_pitstops = (
    spark.read
    .option("multiline", "true")
    .json(f"{bronze_path}/pitstops/pitstops_raw.json")
)

# Check if we have data
if raw_pitstops.count() > 0:
    pitstops_flat = (
        raw_pitstops
        .select(
            col("season"),
            col("round"),
            col("raceName"),
            explode("PitStops").alias("pit")
        )
        .select(
            "season", "round", "raceName",
            col("pit.driverId").alias("driverId"),
            col("pit.lap").alias("lap"),
            col("pit.stop").alias("stopNumber"),
            col("pit.time").alias("timeOfDay"),
            col("pit.duration").alias("duration"),
        )
        .withColumn("season", col("season").cast("int"))
        .withColumn("round", col("round").cast("int"))
        .withColumn("lap", col("lap").cast("int"))
        .withColumn("stopNumber", col("stopNumber").cast("int"))
    )

    print(f"Pit stop records: {pitstops_flat.count()}")

    (
        pitstops_flat.write
        .mode("overwrite")
        .format("delta")
        .partitionBy("season")
        .saveAsTable("f1_bronze.pit_stops")
    )

    print("✓ f1_bronze.pit_stops saved as Delta table")
    display(pitstops_flat.limit(10))
else:
    print("⚠ No pit stop data found — will retry in silver layer")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Lap Times → Delta

# COMMAND ----------

raw_laptimes = (
    spark.read
    .option("multiline", "true")
    .json(f"{bronze_path}/laptimes/laptimes_raw.json")
)

laptimes_df = (
    raw_laptimes
    .withColumn("season", col("season").cast("int"))
    .withColumn("round", col("round").cast("int"))
    .withColumn("lap", col("lap").cast("int"))
    .withColumn("position", col("position").cast("int"))
    .select("season", "round", "raceName", "lap", "driverId", "position", "time")
)

print(f"Lap time records: {laptimes_df.count():,}")

(
    laptimes_df.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_bronze.lap_times")
)

print("✓ f1_bronze.lap_times saved as Delta table")
display(laptimes_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Driver Standings → Delta

# COMMAND ----------

raw_driver_standings = (
    spark.read
    .option("multiline", "true")
    .json(f"{bronze_path}/standings/driver_standings_raw.json")
)

driver_standings_flat = (
    raw_driver_standings
    .select(
        col("season"),
        explode("DriverStandings").alias("standing")
    )
    .select(
        "season",
        col("standing.position").alias("position"),
        col("standing.positionText").alias("positionText"),
        col("standing.points").alias("points"),
        col("standing.wins").alias("wins"),
        col("standing.Driver.driverId").alias("driverId"),
        col("standing.Driver.code").alias("driverCode"),
        col("standing.Driver.givenName").alias("firstName"),
        col("standing.Driver.familyName").alias("lastName"),
        col("standing.Driver.nationality").alias("nationality"),
        col("standing.Constructors").alias("constructors"),
    )
    .withColumn("season", col("season").cast("int"))
    .withColumn("position", col("position").cast("int"))
    .withColumn("points", col("points").cast("float"))
    .withColumn("wins", col("wins").cast("int"))
)

# Extract primary constructor (first in array)
from pyspark.sql.functions import element_at

driver_standings_final = (
    driver_standings_flat
    .withColumn("constructorId", element_at(col("constructors.constructorId"), 1))
    .withColumn("constructorName", element_at(col("constructors.name"), 1))
    .drop("constructors")
)

print(f"Driver standing records: {driver_standings_final.count()}")

(
    driver_standings_final.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_bronze.driver_standings")
)

print("✓ f1_bronze.driver_standings saved as Delta table")
display(driver_standings_final.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Constructor Standings → Delta

# COMMAND ----------

raw_constructor_standings = (
    spark.read
    .option("multiline", "true")
    .json(f"{bronze_path}/standings/constructor_standings_raw.json")
)

constructor_standings_flat = (
    raw_constructor_standings
    .select(
        col("season"),
        explode("ConstructorStandings").alias("standing")
    )
    .select(
        "season",
        col("standing.position").alias("position"),
        col("standing.positionText").alias("positionText"),
        col("standing.points").alias("points"),
        col("standing.wins").alias("wins"),
        col("standing.Constructor.constructorId").alias("constructorId"),
        col("standing.Constructor.name").alias("constructorName"),
        col("standing.Constructor.nationality").alias("nationality"),
    )
    .withColumn("season", col("season").cast("int"))
    .withColumn("position", col("position").cast("int"))
    .withColumn("points", col("points").cast("float"))
    .withColumn("wins", col("wins").cast("int"))
)

print(f"Constructor standing records: {constructor_standings_flat.count()}")

(
    constructor_standings_flat.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_bronze.constructor_standings")
)

print("✓ f1_bronze.constructor_standings saved as Delta table")
display(constructor_standings_flat.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Verify All Delta Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN f1_bronze

# COMMAND ----------

print("=" * 60)
print("  BRONZE DELTA TABLES SUMMARY")
print("=" * 60)

tables = ["race_results", "qualifying", "pit_stops", "lap_times", 
          "driver_standings", "constructor_standings"]

for table in tables:
    try:
        count = spark.sql(f"SELECT COUNT(*) as cnt FROM f1_bronze.{table}").collect()[0]["cnt"]
        print(f"  ✓ f1_bronze.{table:30s} → {count:>8,} rows")
    except Exception as e:
        print(f"  ✗ f1_bronze.{table:30s} → not found")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Quick Analysis — Prove It Works!
# MAGIC 
# MAGIC Let's run a few SQL queries to show the data is real and queryable.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Top 10 drivers by total career points (2018-2025)
# MAGIC SELECT 
# MAGIC     driverCode,
# MAGIC     firstName,
# MAGIC     lastName,
# MAGIC     constructorName,
# MAGIC     ROUND(SUM(points), 1) as total_points,
# MAGIC     COUNT(*) as races,
# MAGIC     SUM(CASE WHEN position = 1 THEN 1 ELSE 0 END) as wins,
# MAGIC     SUM(CASE WHEN position <= 3 THEN 1 ELSE 0 END) as podiums
# MAGIC FROM f1_bronze.race_results
# MAGIC GROUP BY driverCode, firstName, lastName, constructorName
# MAGIC ORDER BY total_points DESC
# MAGIC LIMIT 15

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Average pit stop duration by constructor (2023-2025)
# MAGIC SELECT 
# MAGIC     r.constructorName,
# MAGIC     COUNT(p.stopNumber) as total_stops,
# MAGIC     ROUND(AVG(CAST(p.duration AS FLOAT)), 3) as avg_duration_sec
# MAGIC FROM f1_bronze.pit_stops p
# MAGIC JOIN f1_bronze.race_results r 
# MAGIC     ON p.season = r.season 
# MAGIC     AND p.round = r.round 
# MAGIC     AND p.driverId = r.driverId
# MAGIC WHERE p.season >= 2023
# MAGIC GROUP BY r.constructorName
# MAGIC HAVING COUNT(p.stopNumber) > 10
# MAGIC ORDER BY avg_duration_sec ASC

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Notebook Complete!
# MAGIC 
# MAGIC **What we built:**
# MAGIC - 6 Delta tables in the `f1_bronze` database
# MAGIC - Flattened nested JSON into clean, typed columns
# MAGIC - Partitioned by season for efficient queries
# MAGIC - Verified with real SQL analytics
# MAGIC 
# MAGIC **Key concepts for interviews:**
# MAGIC - **Delta Lake** vs Parquet: ACID transactions, time travel, schema enforcement
# MAGIC - **Partitioning**: why season? Because most queries filter by season
# MAGIC - **Explode**: how to flatten nested JSON arrays in Spark
# MAGIC 
# MAGIC **Next:** `03_silver_layer` — Clean, enrich, and join tables
