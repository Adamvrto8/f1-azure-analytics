# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook 03 — Silver Layer
# MAGIC 
# MAGIC Transform bronze data into clean, enriched, joined datasets:
# MAGIC 1. **Clean**: remove nulls, fix types, standardize formats
# MAGIC 2. **Enrich**: add calculated columns (time in seconds, gaps, flags)
# MAGIC 3. **Join**: combine results + qualifying + pit stops into unified race views
# MAGIC 
# MAGIC Silver = trusted, analytics-ready data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Setup

# COMMAND ----------

storage_account_name = "f1analyticslake2"

# >>> PASTE YOUR KEY BELOW <<<
storage_account_key = "PASTE_YOUR_KEY_HERE"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net",
    storage_account_key
)

silver_path = f"abfss://silver@{storage_account_name}.dfs.core.windows.net"

spark.sql("CREATE DATABASE IF NOT EXISTS f1_silver")
spark.sql("USE f1_silver")

print("✓ Storage connected, f1_silver database ready")

# COMMAND ----------

from pyspark.sql.functions import (
    col, expr, when, lit, concat, round as spark_round,
    regexp_extract, split, element_at, lag, sum as spark_sum,
    count, avg, min as spark_min, max as spark_max, row_number, dense_rank
)
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Silver Race Results
# MAGIC 
# MAGIC Clean results + add computed columns

# COMMAND ----------

bronze_results = spark.table("f1_bronze.race_results")

silver_results = (
    bronze_results
    # Clean: fill missing positions with 99 (DNF/DNS)
    .withColumn("position", when(col("position").isNull(), 99).otherwise(col("position")))
    .withColumn("gridPosition", when(col("gridPosition").isNull(), 0).otherwise(col("gridPosition")))
    .withColumn("points", when(col("points").isNull(), 0.0).otherwise(col("points")))
    
    # Enrich: full driver name
    .withColumn("driverName", concat(col("firstName"), lit(" "), col("lastName")))
    
    # Enrich: positions gained/lost from grid to finish
    .withColumn("positionsGained", 
        when((col("gridPosition") > 0) & (col("position") < 99),
             col("gridPosition") - col("position"))
        .otherwise(lit(None)))
    
    # Enrich: did the driver finish?
    .withColumn("finished", 
        when(col("status") == "Finished", lit(True))
        .when(col("status").startswith("+"), lit(True))  # "+1 Lap" etc
        .otherwise(lit(False)))
    
    # Enrich: is this a podium finish?
    .withColumn("isPodium", col("position") <= 3)
    
    # Enrich: is this a win?
    .withColumn("isWin", col("position") == 1)
    
    # Enrich: is this a points finish? (top 10 in modern F1)
    .withColumn("isPointsFinish", col("points") > 0)
    
    # Drop raw name columns (we have driverName now)
    .drop("firstName", "lastName")
)

print(f"Silver race results: {silver_results.count():,} rows")
print(f"Columns: {len(silver_results.columns)}")

(
    silver_results.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_silver.race_results")
)

print("✓ f1_silver.race_results saved")

# COMMAND ----------

# Quick check: who gained the most positions in a single race?
display(
    silver_results
    .filter(col("positionsGained").isNotNull())
    .orderBy(col("positionsGained").desc())
    .select("season", "raceName", "driverName", "constructorName", 
            "gridPosition", "position", "positionsGained")
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Silver Qualifying
# MAGIC 
# MAGIC Parse qualifying times into seconds for comparison

# COMMAND ----------

bronze_qualifying = spark.table("f1_bronze.qualifying")

def time_to_seconds_expr(time_col):
    """Convert 'M:SS.mmm' format to seconds as float."""
    return (
        expr(f"try_cast(split({time_col}, ':')[0] as float)") * 60 +
        expr(f"try_cast(split({time_col}, ':')[1] as float)")
    )

silver_qualifying = (
    bronze_qualifying
    # Parse Q1, Q2, Q3 times to seconds
    .withColumn("q1Seconds", 
        when(col("q1Time").isNotNull() & (col("q1Time") != ""),
             time_to_seconds_expr("q1Time"))
        .otherwise(lit(None)))
    .withColumn("q2Seconds",
        when(col("q2Time").isNotNull() & (col("q2Time") != ""),
             time_to_seconds_expr("q2Time"))
        .otherwise(lit(None)))
    .withColumn("q3Seconds",
        when(col("q3Time").isNotNull() & (col("q3Time") != ""),
             time_to_seconds_expr("q3Time"))
        .otherwise(lit(None)))
    
    # Best qualifying time (Q3 if available, else Q2, else Q1)
    .withColumn("bestQualTime",
        when(col("q3Seconds").isNotNull(), col("q3Seconds"))
        .when(col("q2Seconds").isNotNull(), col("q2Seconds"))
        .when(col("q1Seconds").isNotNull(), col("q1Seconds"))
        .otherwise(lit(None)))
    
    # Full driver name
    .withColumn("driverName", concat(col("firstName"), lit(" "), col("lastName")))
    .drop("firstName", "lastName")
)

# Add gap to pole (difference from P1 time per race)
pole_window = Window.partitionBy("season", "round").orderBy("position")

silver_qualifying = (
    silver_qualifying
    .withColumn("poleTime", 
        expr("FIRST_VALUE(bestQualTime) OVER (PARTITION BY season, round ORDER BY position)"))
    .withColumn("gapToPole",
        when(col("bestQualTime").isNotNull() & col("poleTime").isNotNull(),
             spark_round(col("bestQualTime") - col("poleTime"), 3))
        .otherwise(lit(None)))
)

print(f"Silver qualifying: {silver_qualifying.count():,} rows")

(
    silver_qualifying.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_silver.qualifying")
)

print("✓ f1_silver.qualifying saved")

# COMMAND ----------

# Check: average gap to pole by constructor
display(
    silver_qualifying
    .filter(col("gapToPole").isNotNull() & (col("season") >= 2022))
    .groupBy("constructorName")
    .agg(
        spark_round(avg("gapToPole"), 3).alias("avgGapToPole"),
        count("*").alias("qualSessions")
    )
    .orderBy("avgGapToPole")
    .limit(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Silver Pit Stops
# MAGIC 
# MAGIC Parse duration to seconds, flag slow stops

# COMMAND ----------

bronze_pitstops = spark.table("f1_bronze.pit_stops")

silver_pitstops = (
    bronze_pitstops
    # Parse duration string to float seconds
    .withColumn("durationSeconds", expr("try_cast(duration as float)"))
    
    # Flag slow pit stops (> 5 seconds is slow in modern F1)
    .withColumn("isSlowStop", col("durationSeconds") > 5.0)
    
    # Flag very fast pit stops (< 2.5 seconds is elite)
    .withColumn("isFastStop", col("durationSeconds") < 2.5)
)

print(f"Silver pit stops: {silver_pitstops.count():,} rows")

(
    silver_pitstops.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_silver.pit_stops")
)

print("✓ f1_silver.pit_stops saved")

# COMMAND ----------

# Check: fastest pit stops ever
display(
    silver_pitstops
    .filter(col("durationSeconds") > 0)
    .orderBy("durationSeconds")
    .select("season", "round", "raceName", "driverId", "stopNumber", 
            "lap", "durationSeconds")
    .limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Silver Lap Times
# MAGIC 
# MAGIC Parse lap time strings to seconds, calculate deltas

# COMMAND ----------

bronze_laptimes = spark.table("f1_bronze.lap_times")

silver_laptimes = (
    bronze_laptimes
    # Parse "M:SS.mmm" to seconds
    .withColumn("timeSeconds",
        when(col("time").isNotNull() & (col("time") != ""),
             expr("try_cast(split(time, ':')[0] as float)") * 60 +
             expr("try_cast(split(time, ':')[1] as float)"))
        .otherwise(lit(None)))
)

# Add delta to race leader's lap time per lap
leader_window = Window.partitionBy("season", "round", "lap").orderBy("position")

silver_laptimes = (
    silver_laptimes
    .withColumn("leaderTime",
        expr("FIRST_VALUE(timeSeconds) OVER (PARTITION BY season, round, lap ORDER BY position)"))
    .withColumn("gapToLeader",
        when(col("timeSeconds").isNotNull() & col("leaderTime").isNotNull(),
             spark_round(col("timeSeconds") - col("leaderTime"), 3))
        .otherwise(lit(None)))
)

# Add lap time delta (difference from driver's previous lap)
driver_lap_window = Window.partitionBy("season", "round", "driverId").orderBy("lap")

silver_laptimes = (
    silver_laptimes
    .withColumn("prevLapTime", lag("timeSeconds", 1).over(driver_lap_window))
    .withColumn("lapTimeDelta",
        when(col("timeSeconds").isNotNull() & col("prevLapTime").isNotNull(),
             spark_round(col("timeSeconds") - col("prevLapTime"), 3))
        .otherwise(lit(None)))
    .drop("prevLapTime")
)

print(f"Silver lap times: {silver_laptimes.count():,} rows")

(
    silver_laptimes.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_silver.lap_times")
)

print("✓ f1_silver.lap_times saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Silver Driver Standings (clean)

# COMMAND ----------

bronze_driver_standings = spark.table("f1_bronze.driver_standings")

silver_driver_standings = (
    bronze_driver_standings
    .withColumn("driverName", concat(col("firstName"), lit(" "), col("lastName")))
    .drop("firstName", "lastName")
)

(
    silver_driver_standings.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_silver.driver_standings")
)

print(f"✓ f1_silver.driver_standings saved ({silver_driver_standings.count()} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Silver Constructor Standings (clean)

# COMMAND ----------

bronze_constructor_standings = spark.table("f1_bronze.constructor_standings")

(
    bronze_constructor_standings.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_silver.constructor_standings")
)

print(f"✓ f1_silver.constructor_standings saved ({bronze_constructor_standings.count()} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Joined Race View (the star of the silver layer)
# MAGIC 
# MAGIC Combine race results + qualifying + pit stops into one unified table.
# MAGIC This is the table you'll use most in the gold layer and for dashboards.

# COMMAND ----------

# Get qualifying data we need
qual_for_join = (
    spark.table("f1_silver.qualifying")
    .select(
        col("season").alias("q_season"),
        col("round").alias("q_round"),
        col("driverId").alias("q_driverId"),
        "q1Seconds", "q2Seconds", "q3Seconds", 
        "bestQualTime", "gapToPole"
    )
)

# Get pit stop summary per driver per race
pitstop_summary = (
    spark.table("f1_silver.pit_stops")
    .groupBy(
        col("season").alias("p_season"),
        col("round").alias("p_round"),
        col("driverId").alias("p_driverId")
    )
    .agg(
        count("*").alias("numPitStops"),
        spark_round(spark_min("durationSeconds"), 3).alias("fastestPitStop"),
        spark_round(avg("durationSeconds"), 3).alias("avgPitStopDuration"),
        spark_round(spark_sum("durationSeconds"), 3).alias("totalPitStopTime")
    )
)

# Join everything together
race_view = (
    spark.table("f1_silver.race_results")
    # Join qualifying
    .join(
        qual_for_join,
        (col("season") == col("q_season")) & 
        (col("round") == col("q_round")) & 
        (col("driverId") == col("q_driverId")),
        "left"
    )
    .drop("q_season", "q_round", "q_driverId")
    # Join pit stop summary
    .join(
        pitstop_summary,
        (col("season") == col("p_season")) & 
        (col("round") == col("p_round")) & 
        (col("driverId") == col("p_driverId")),
        "left"
    )
    .drop("p_season", "p_round", "p_driverId")
    # Fill nulls for pit stops (some races have no pit data)
    .withColumn("numPitStops", when(col("numPitStops").isNull(), 0).otherwise(col("numPitStops")))
)

print(f"Joined race view: {race_view.count():,} rows, {len(race_view.columns)} columns")

(
    race_view.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_silver.race_view")
)

print("✓ f1_silver.race_view saved — this is your main analytics table!")

# COMMAND ----------

# Preview the joined data
display(
    race_view
    .filter((col("season") == 2024) & (col("round") == 1))
    .orderBy("position")
    .select("position", "driverName", "constructorName", "gridPosition",
            "positionsGained", "points", "bestQualTime", "gapToPole",
            "numPitStops", "avgPitStopDuration", "status")
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Verify Silver Layer

# COMMAND ----------

print("=" * 60)
print("  SILVER LAYER SUMMARY")
print("=" * 60)

silver_tables = [
    "race_results", "qualifying", "pit_stops", "lap_times",
    "driver_standings", "constructor_standings", "race_view"
]

for table in silver_tables:
    try:
        count = spark.sql(f"SELECT COUNT(*) as cnt FROM f1_silver.{table}").collect()[0]["cnt"]
        cols = len(spark.table(f"f1_silver.{table}").columns)
        print(f"  ✓ f1_silver.{table:30s} → {count:>8,} rows  ({cols} cols)")
    except Exception as e:
        print(f"  ✗ f1_silver.{table:30s} → error: {e}")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Silver Layer Complete!
# MAGIC 
# MAGIC **What we built:**
# MAGIC - Cleaned & typed all bronze tables
# MAGIC - Parsed time strings (qualifying, lap times, pit stops) to numeric seconds
# MAGIC - Added computed columns: positionsGained, gapToPole, lapTimeDelta, flags
# MAGIC - Created `race_view` — a joined table combining results + qualifying + pit stops
# MAGIC 
# MAGIC **Key concepts for interviews:**
# MAGIC - **Silver layer purpose**: trusted, cleaned, enriched data ready for analytics
# MAGIC - **Window functions**: FIRST_VALUE, LAG for computing gaps and deltas
# MAGIC - **Star schema thinking**: race_view is a denormalized fact table
# MAGIC - **try_cast**: handling dirty data gracefully without pipeline failures
# MAGIC 
# MAGIC **Next:** `04_gold_layer` — Aggregated business-level views and ML features
