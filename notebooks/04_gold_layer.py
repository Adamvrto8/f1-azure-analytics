# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook 04 — Gold Layer
# MAGIC 
# MAGIC Pre-aggregated, business-ready tables for dashboards and ML:
# MAGIC 1. **Driver season stats** — wins, podiums, points, consistency
# MAGIC 2. **Constructor season stats** — team performance comparison
# MAGIC 3. **Race strategy analysis** — pit stop patterns and their impact
# MAGIC 4. **Driver race performance** — qualifying vs race pace, overtaking
# MAGIC 5. **Head-to-head teammates** — who beats their teammate more?
# MAGIC 
# MAGIC Gold = ready for Power BI dashboards and ML models

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

gold_path = f"abfss://gold@{storage_account_name}.dfs.core.windows.net"

spark.sql("CREATE DATABASE IF NOT EXISTS f1_gold")
spark.sql("USE f1_gold")

print("✓ Storage connected, f1_gold database ready")

# COMMAND ----------

from pyspark.sql.functions import (
    col, expr, when, lit, concat, round as spark_round,
    count, avg, min as spark_min, max as spark_max, sum as spark_sum,
    first, last, collect_list, stddev, percentile_approx,
    row_number, dense_rank, percent_rank
)
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Driver Season Stats
# MAGIC 
# MAGIC One row per driver per season — everything an interviewer would ask about.

# COMMAND ----------

race_view = spark.table("f1_silver.race_view")

driver_season = (
    race_view
    .groupBy("season", "driverId", "driverCode", "driverName", "constructorId", "constructorName")
    .agg(
        # Race counts
        count("*").alias("races"),
        spark_sum(when(col("isWin"), 1).otherwise(0)).alias("wins"),
        spark_sum(when(col("isPodium"), 1).otherwise(0)).alias("podiums"),
        spark_sum(when(col("isPointsFinish"), 1).otherwise(0)).alias("pointsFinishes"),
        spark_sum(when(col("finished"), 1).otherwise(0)).alias("finishes"),
        spark_sum(when(~col("finished"), 1).otherwise(0)).alias("retirements"),
        
        # Points
        spark_round(spark_sum("points"), 1).alias("totalPoints"),
        spark_round(avg("points"), 2).alias("avgPointsPerRace"),
        spark_round(spark_max("points"), 1).alias("bestRacePoints"),
        
        # Positions
        spark_round(avg(when(col("position") < 99, col("position"))), 2).alias("avgFinishPosition"),
        spark_min(when(col("position") < 99, col("position"))).alias("bestFinish"),
        spark_round(avg("gridPosition"), 2).alias("avgGridPosition"),
        
        # Overtaking
        spark_round(avg("positionsGained"), 2).alias("avgPositionsGained"),
        spark_sum(when(col("positionsGained") > 0, 1).otherwise(0)).alias("racesWithOvertakes"),
        spark_max("positionsGained").alias("bestOvertakeRace"),
        
        # Qualifying
        spark_round(avg("bestQualTime"), 3).alias("avgQualTime"),
        spark_round(avg("gapToPole"), 3).alias("avgGapToPole"),
        spark_min("gapToPole").alias("closestToPole"),
        spark_sum(when(col("gridPosition") == 1, 1).otherwise(0)).alias("poles"),
        
        # Pit stops
        spark_round(avg("avgPitStopDuration"), 3).alias("avgPitStopTime"),
        spark_round(spark_min("fastestPitStop"), 3).alias("bestPitStop"),
    )
    # Derived metrics
    .withColumn("finishRate", spark_round(col("finishes") / col("races") * 100, 1))
    .withColumn("podiumRate", spark_round(col("podiums") / col("races") * 100, 1))
    .withColumn("winRate", spark_round(col("wins") / col("races") * 100, 1))
    .orderBy("season", col("totalPoints").desc())
)

print(f"Driver season stats: {driver_season.count()} rows")

(
    driver_season.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_gold.driver_season_stats")
)

print("✓ f1_gold.driver_season_stats saved")

# COMMAND ----------

# Preview: 2024 season standings
display(
    driver_season
    .filter(col("season") == 2024)
    .orderBy(col("totalPoints").desc())
    .select("driverName", "constructorName", "races", "wins", "podiums", 
            "totalPoints", "avgFinishPosition", "avgGapToPole", "finishRate")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Constructor Season Stats

# COMMAND ----------

constructor_season = (
    race_view
    .groupBy("season", "constructorId", "constructorName")
    .agg(
        count("*").alias("raceEntries"),
        spark_sum(when(col("isWin"), 1).otherwise(0)).alias("wins"),
        spark_sum(when(col("isPodium"), 1).otherwise(0)).alias("podiums"),
        spark_round(spark_sum("points"), 1).alias("totalPoints"),
        spark_round(avg("points"), 2).alias("avgPointsPerEntry"),
        spark_round(avg(when(col("position") < 99, col("position"))), 2).alias("avgFinishPosition"),
        spark_round(avg("gridPosition"), 2).alias("avgGridPosition"),
        spark_round(avg("positionsGained"), 2).alias("avgPositionsGained"),
        spark_round(avg("bestQualTime"), 3).alias("avgQualTime"),
        spark_round(avg("gapToPole"), 3).alias("avgGapToPole"),
        spark_round(avg("avgPitStopDuration"), 3).alias("avgPitStopTime"),
        spark_sum(when(col("finished"), 1).otherwise(0)).alias("finishes"),
        spark_sum(when(~col("finished"), 1).otherwise(0)).alias("retirements"),
    )
    .withColumn("reliabilityRate", spark_round(col("finishes") / col("raceEntries") * 100, 1))
    .withColumn("winRate", spark_round(col("wins") / col("raceEntries") * 100, 1))
    .orderBy("season", col("totalPoints").desc())
)

print(f"Constructor season stats: {constructor_season.count()} rows")

(
    constructor_season.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_gold.constructor_season_stats")
)

print("✓ f1_gold.constructor_season_stats saved")

# COMMAND ----------

# Preview: constructor performance trend
display(
    constructor_season
    .filter(col("constructorName").isin("Red Bull", "Ferrari", "Mercedes", "McLaren"))
    .select("season", "constructorName", "totalPoints", "wins", "avgFinishPosition", "avgGapToPole")
    .orderBy("constructorName", "season")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Race Strategy Analysis
# MAGIC 
# MAGIC How do pit stop strategies correlate with race outcomes?

# COMMAND ----------

strategy_analysis = (
    race_view
    .filter(col("finished") & col("numPitStops") > 0)
    .withColumn("strategyType",
        when(col("numPitStops") == 1, "1-stop")
        .when(col("numPitStops") == 2, "2-stop")
        .when(col("numPitStops") == 3, "3-stop")
        .otherwise("4+ stops"))
    .groupBy("season", "raceName", "round", "strategyType")
    .agg(
        count("*").alias("driversUsing"),
        spark_round(avg(when(col("position") < 99, col("position"))), 2).alias("avgFinishPosition"),
        spark_round(avg("points"), 2).alias("avgPoints"),
        spark_round(avg("positionsGained"), 2).alias("avgPositionsGained"),
        spark_round(avg("totalPitStopTime"), 2).alias("avgTotalPitTime"),
        spark_sum(when(col("isPodium"), 1).otherwise(0)).alias("podiums"),
        spark_sum(when(col("isWin"), 1).otherwise(0)).alias("wins"),
    )
    .orderBy("season", "round", "strategyType")
)

print(f"Strategy analysis: {strategy_analysis.count()} rows")

(
    strategy_analysis.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_gold.race_strategy_analysis")
)

print("✓ f1_gold.race_strategy_analysis saved")

# COMMAND ----------

# Which strategy wins more often?
display(
    strategy_analysis
    .groupBy("strategyType")
    .agg(
        count("*").alias("raceInstances"),
        spark_round(avg("avgFinishPosition"), 2).alias("overallAvgPosition"),
        spark_round(avg("avgPoints"), 2).alias("overallAvgPoints"),
        spark_sum("wins").alias("totalWins"),
        spark_sum("podiums").alias("totalPodiums"),
    )
    .orderBy("overallAvgPosition")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Driver Race Performance Detail
# MAGIC 
# MAGIC Per-driver per-race metrics: qualifying pace vs race pace, consistency.

# COMMAND ----------

# Get average lap time per driver per race from silver lap times
avg_laptimes = (
    spark.table("f1_silver.lap_times")
    .filter(col("timeSeconds").isNotNull() & (col("timeSeconds") > 0))
    .groupBy(
        col("season").alias("lt_season"),
        col("round").alias("lt_round"),
        col("driverId").alias("lt_driverId")
    )
    .agg(
        spark_round(avg("timeSeconds"), 3).alias("avgLapTime"),
        spark_round(spark_min("timeSeconds"), 3).alias("fastestLap"),
        spark_round(stddev("timeSeconds"), 3).alias("lapTimeStdDev"),
        count("*").alias("lapsCompleted"),
        spark_round(avg("gapToLeader"), 3).alias("avgGapToLeader"),
    )
)

# Join with race view
driver_race_perf = (
    race_view
    .join(
        avg_laptimes,
        (col("season") == col("lt_season")) &
        (col("round") == col("lt_round")) &
        (col("driverId") == col("lt_driverId")),
        "left"
    )
    .drop("lt_season", "lt_round", "lt_driverId")
    .select(
        "season", "round", "raceName", "circuitId", "circuitName",
        "driverId", "driverCode", "driverName", "constructorName",
        "gridPosition", "position", "positionsGained", "points",
        "finished", "status",
        "bestQualTime", "gapToPole",
        "numPitStops", "avgPitStopDuration", "totalPitStopTime",
        "avgLapTime", "fastestLap", "lapTimeStdDev", "avgGapToLeader"
    )
    # Consistency score: lower stddev = more consistent
    .withColumn("consistencyScore",
        when(col("lapTimeStdDev").isNotNull() & (col("lapTimeStdDev") > 0),
             spark_round(1 / col("lapTimeStdDev") * 100, 2))
        .otherwise(lit(None)))
)

print(f"Driver race performance: {driver_race_perf.count():,} rows")

(
    driver_race_perf.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_gold.driver_race_performance")
)

print("✓ f1_gold.driver_race_performance saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Teammate Head-to-Head
# MAGIC 
# MAGIC Compare drivers within the same team — who beats whom?

# COMMAND ----------

# Get pairs of teammates per race
teammates_base = (
    race_view
    .select("season", "round", "raceName", "constructorId", "constructorName",
            "driverId", "driverCode", "driverName", "position", "gridPosition",
            "points", "bestQualTime", "finished")
)

# Self-join to get teammate pairs
from pyspark.sql.functions import least, greatest

teammate_pairs = (
    teammates_base.alias("d1")
    .join(
        teammates_base.alias("d2"),
        (col("d1.season") == col("d2.season")) &
        (col("d1.round") == col("d2.round")) &
        (col("d1.constructorId") == col("d2.constructorId")) &
        (col("d1.driverId") < col("d2.driverId")),  # avoid duplicates
        "inner"
    )
    .select(
        col("d1.season").alias("season"),
        col("d1.round").alias("round"),
        col("d1.raceName").alias("raceName"),
        col("d1.constructorName").alias("constructorName"),
        col("d1.driverCode").alias("driver1Code"),
        col("d1.driverName").alias("driver1Name"),
        col("d1.position").alias("driver1Pos"),
        col("d1.points").alias("driver1Points"),
        col("d1.bestQualTime").alias("driver1QualTime"),
        col("d2.driverCode").alias("driver2Code"),
        col("d2.driverName").alias("driver2Name"),
        col("d2.position").alias("driver2Pos"),
        col("d2.points").alias("driver2Points"),
        col("d2.bestQualTime").alias("driver2QualTime"),
    )
    .withColumn("raceWinner",
        when(col("driver1Pos") < col("driver2Pos"), col("driver1Code"))
        .when(col("driver2Pos") < col("driver1Pos"), col("driver2Code"))
        .otherwise(lit("tie")))
    .withColumn("qualWinner",
        when(col("driver1QualTime").isNotNull() & col("driver2QualTime").isNotNull(),
            when(col("driver1QualTime") < col("driver2QualTime"), col("driver1Code"))
            .when(col("driver2QualTime") < col("driver1QualTime"), col("driver2Code"))
            .otherwise(lit("tie")))
        .otherwise(lit(None)))
)

# Aggregate head-to-head by season
h2h_season = (
    teammate_pairs
    .groupBy("season", "constructorName", "driver1Code", "driver1Name", "driver2Code", "driver2Name")
    .agg(
        count("*").alias("racesCompared"),
        spark_sum(when(col("raceWinner") == col("driver1Code"), 1).otherwise(0)).alias("driver1RaceWins"),
        spark_sum(when(col("raceWinner") == col("driver2Code"), 1).otherwise(0)).alias("driver2RaceWins"),
        spark_sum(when(col("qualWinner") == col("driver1Code"), 1).otherwise(0)).alias("driver1QualWins"),
        spark_sum(when(col("qualWinner") == col("driver2Code"), 1).otherwise(0)).alias("driver2QualWins"),
        spark_round(spark_sum(col("driver1Points")), 1).alias("driver1TotalPoints"),
        spark_round(spark_sum(col("driver2Points")), 1).alias("driver2TotalPoints"),
    )
    .orderBy("season", "constructorName")
)

print(f"Teammate head-to-head: {h2h_season.count()} rows")

(
    h2h_season.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_gold.teammate_head_to_head")
)

print("✓ f1_gold.teammate_head_to_head saved")

# COMMAND ----------

# Preview: 2024 teammate battles
display(
    h2h_season
    .filter(col("season") == 2024)
    .orderBy(col("driver1TotalPoints").desc())
    .select("constructorName", "driver1Code", "driver2Code", 
            "racesCompared", "driver1RaceWins", "driver2RaceWins",
            "driver1QualWins", "driver2QualWins",
            "driver1TotalPoints", "driver2TotalPoints")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Circuit Performance Profile
# MAGIC 
# MAGIC How do teams and drivers perform at specific circuits?

# COMMAND ----------

circuit_profile = (
    race_view
    .filter(col("position") < 99)
    .groupBy("circuitId", "circuitName", "country", "constructorName")
    .agg(
        count("*").alias("raceEntries"),
        spark_round(avg("position"), 2).alias("avgFinish"),
        spark_round(avg("gridPosition"), 2).alias("avgGrid"),
        spark_round(avg("points"), 2).alias("avgPoints"),
        spark_sum(when(col("isWin"), 1).otherwise(0)).alias("wins"),
        spark_sum(when(col("isPodium"), 1).otherwise(0)).alias("podiums"),
        spark_round(avg("gapToPole"), 3).alias("avgGapToPole"),
    )
    .filter(col("raceEntries") >= 3)  # At least 3 entries for meaningful stats
    .orderBy("circuitId", "avgFinish")
)

print(f"Circuit performance profiles: {circuit_profile.count()} rows")

(
    circuit_profile.write
    .mode("overwrite")
    .format("delta")
    .saveAsTable("f1_gold.circuit_performance")
)

print("✓ f1_gold.circuit_performance saved")

# COMMAND ----------

# Which teams dominate which circuits?
display(
    circuit_profile
    .filter(col("wins") > 0)
    .orderBy(col("wins").desc(), "avgFinish")
    .select("circuitName", "country", "constructorName", "raceEntries", 
            "wins", "podiums", "avgFinish", "avgGapToPole")
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Points Progression (for dashboard line charts)
# MAGIC 
# MAGIC Cumulative points per driver per race through a season.

# COMMAND ----------

points_progression = (
    race_view
    .select("season", "round", "raceName", "driverId", "driverCode", 
            "driverName", "constructorName", "points", "position")
    .withColumn("cumulativePoints",
        spark_sum("points").over(
            Window.partitionBy("season", "driverId")
            .orderBy("round")
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        ))
    .withColumn("cumulativePoints", spark_round("cumulativePoints", 1))
    .orderBy("season", "driverId", "round")
)

print(f"Points progression: {points_progression.count():,} rows")

(
    points_progression.write
    .mode("overwrite")
    .format("delta")
    .partitionBy("season")
    .saveAsTable("f1_gold.points_progression")
)

print("✓ f1_gold.points_progression saved")

# COMMAND ----------

# Preview: 2024 title fight progression
display(
    points_progression
    .filter(
        (col("season") == 2024) & 
        col("driverCode").isin("VER", "NOR", "LEC", "PIA", "SAI", "HAM")
    )
    .select("round", "raceName", "driverCode", "points", "cumulativePoints")
    .orderBy("driverCode", "round")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Verify Gold Layer

# COMMAND ----------

print("=" * 65)
print("  GOLD LAYER SUMMARY")
print("=" * 65)

gold_tables = [
    "driver_season_stats", "constructor_season_stats", 
    "race_strategy_analysis", "driver_race_performance",
    "teammate_head_to_head", "circuit_performance",
    "points_progression"
]

for table in gold_tables:
    try:
        count = spark.sql(f"SELECT COUNT(*) as cnt FROM f1_gold.{table}").collect()[0]["cnt"]
        cols = len(spark.table(f"f1_gold.{table}").columns)
        print(f"  ✓ f1_gold.{table:30s} → {count:>8,} rows  ({cols} cols)")
    except Exception as e:
        print(f"  ✗ f1_gold.{table:30s} → error")

print("=" * 65)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Gold Layer Complete!
# MAGIC 
# MAGIC **Tables built:**
# MAGIC | Table | Purpose | Dashboard Use |
# MAGIC |-------|---------|--------------|
# MAGIC | driver_season_stats | Driver performance per season | Season overview, driver comparison |
# MAGIC | constructor_season_stats | Team performance per season | Constructor championship analysis |
# MAGIC | race_strategy_analysis | Pit strategy vs outcomes | Strategy deep-dive |
# MAGIC | driver_race_performance | Per-race metrics with lap data | Race analysis, consistency tracking |
# MAGIC | teammate_head_to_head | Intra-team driver comparison | Teammate battles |
# MAGIC | circuit_performance | Team/driver strength by track | Circuit-specific analysis |
# MAGIC | points_progression | Cumulative points through season | Championship line charts |
# MAGIC 
# MAGIC **Key concepts for interviews:**
# MAGIC - **Gold layer purpose**: pre-aggregated for specific business questions
# MAGIC - **Window functions**: cumulative sums, ranking, lag/lead
# MAGIC - **Self-joins**: teammate comparison pattern
# MAGIC - **Denormalization**: trading storage for query speed
# MAGIC 
# MAGIC **Next steps:**
# MAGIC - Connect Power BI to these tables
# MAGIC - Build ML models (podium predictor, strategy optimizer)
# MAGIC - Set up Azure Data Factory pipeline to refresh all layers
