# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook 06 — F1 Data Visualization & Analysis
# MAGIC 
# MAGIC Interactive visualizations built from the gold layer tables.
# MAGIC Explore driver performance, team battles, strategies, and trends.
# MAGIC 
# MAGIC Sections:
# MAGIC 1. Championship battles — who dominated which era?
# MAGIC 2. Constructor power shifts — rise and fall of teams
# MAGIC 3. Qualifying vs race performance — who converts poles to wins?
# MAGIC 4. Pit stop analysis — which teams have the fastest crews?
# MAGIC 5. Overtaking kings — who gains the most positions?
# MAGIC 6. Teammate battles — intra-team rivalry
# MAGIC 7. Circuit specialists — who dominates where?
# MAGIC 8. Strategy deep-dive — does 1-stop or 2-stop win more?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

storage_account_name = "f1analyticslake2"

# >>> PASTE YOUR KEY BELOW <<<
storage_account_key = "PASTE_YOUR_KEY_HERE"

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net",
    storage_account_key
)

print("✓ Storage connected")

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

# Consistent style
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# F1 team colors (approximate)
TEAM_COLORS = {
    "Red Bull": "#3671C6",
    "Mercedes": "#6CD3BF",
    "Ferrari": "#F91536",
    "McLaren": "#F58020",
    "Aston Martin": "#358C75",
    "Alpine F1 Team": "#2293D1",
    "AlphaTauri": "#5E8FAA",
    "RB F1 Team": "#6692FF",
    "Williams": "#37BEDD",
    "Haas F1 Team": "#B6BABD",
    "Alfa Romeo": "#C92D4B",
    "Racing Point": "#F596C8",
    "Renault": "#FFF500",
    "Toro Rosso": "#469BFF",
    "Sauber": "#52E252",
}

def get_color(team):
    return TEAM_COLORS.get(team, "#888888")

# COMMAND ----------

# Load all gold tables
driver_stats = spark.table("f1_gold.driver_season_stats").toPandas()
constructor_stats = spark.table("f1_gold.constructor_season_stats").toPandas()
points_prog = spark.table("f1_gold.points_progression").toPandas()
strategy = spark.table("f1_gold.race_strategy_analysis").toPandas()
h2h = spark.table("f1_gold.teammate_head_to_head").toPandas()
circuit = spark.table("f1_gold.circuit_performance").toPandas()
race_perf = spark.table("f1_gold.driver_race_performance").toPandas()

print("✓ All gold tables loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Championship battles — points progression
# MAGIC 
# MAGIC Who led the championship at each stage of the season?

# COMMAND ----------

# Pick a season to visualize
SEASON = 2024

season_prog = points_prog[points_prog["season"] == SEASON].copy()

# Get top 6 drivers by final points
final_round = season_prog["round"].max()
final_standings = (
    season_prog[season_prog["round"] == final_round]
    .nlargest(6, "cumulativePoints")
)
top_drivers = final_standings["driverCode"].tolist()

fig, ax = plt.subplots(figsize=(14, 7))

for driver in top_drivers:
    driver_data = season_prog[season_prog["driverCode"] == driver].sort_values("round")
    team = driver_data["constructorName"].iloc[0]
    ax.plot(
        driver_data["round"], driver_data["cumulativePoints"],
        marker="o", markersize=4, linewidth=2.5,
        color=get_color(team), label=f"{driver} ({team})"
    )

ax.set_xlabel("Round")
ax.set_ylabel("Cumulative points")
ax.set_title(f"{SEASON} Championship battle — top 6 drivers")
ax.legend(loc="upper left", framealpha=0.9)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Constructor power shifts across seasons
# MAGIC 
# MAGIC How have the top teams' points evolved year over year?

# COMMAND ----------

top_teams = ["Red Bull", "Mercedes", "Ferrari", "McLaren", "Aston Martin"]

fig, ax = plt.subplots(figsize=(14, 6))

for team in top_teams:
    team_data = (
        constructor_stats[constructor_stats["constructorName"] == team]
        .sort_values("season")
    )
    ax.plot(
        team_data["season"], team_data["totalPoints"],
        marker="o", markersize=6, linewidth=2.5,
        color=get_color(team), label=team
    )

ax.set_xlabel("Season")
ax.set_ylabel("Total points")
ax.set_title("Constructor championship points by season")
ax.legend(framealpha=0.9)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Qualifying vs race — who converts poles to wins?

# COMMAND ----------

# Scatter: average grid position vs average finish position per driver-season
recent = driver_stats[driver_stats["season"] >= 2022].copy()
recent = recent[recent["races"] >= 10]  # At least 10 races

fig, ax = plt.subplots(figsize=(10, 10))

for _, row in recent.iterrows():
    ax.scatter(
        row["avgGridPosition"], row["avgFinishPosition"],
        color=get_color(row["constructorName"]),
        s=row["totalPoints"] * 0.8 + 20,
        alpha=0.7, edgecolors="white", linewidth=0.5
    )
    ax.annotate(
        f"{row['driverCode']} {int(row['season'])}",
        (row["avgGridPosition"], row["avgFinishPosition"]),
        fontsize=8, ha="center", va="bottom",
        xytext=(0, 5), textcoords="offset points"
    )

# Diagonal line: above = gained positions, below = lost positions
lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([0, lim], [0, lim], "k--", alpha=0.2, label="No change line")

ax.set_xlabel("Average grid position (qualifying)")
ax.set_ylabel("Average finish position (race)")
ax.set_title("Qualifying vs race performance (2022-2025)\nBubble size = total points")
ax.invert_xaxis()
ax.invert_yaxis()
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Reading this chart:** drivers above the diagonal line tend to lose positions 
# MAGIC from qualifying to race (poor race pace or reliability). Drivers below the line 
# MAGIC gain positions (strong racers, good strategy). Bubble size shows total points.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Pit stop performance by team
# MAGIC 
# MAGIC Which teams have the fastest pit crews?

# COMMAND ----------

pit_data = spark.table("f1_silver.pit_stops").toPandas()
results_data = spark.table("f1_silver.race_results").toPandas()

# Join pit stops with results to get constructor
pit_merged = pit_data.merge(
    results_data[["season", "round", "driverId", "constructorName"]],
    on=["season", "round", "driverId"],
    how="left"
)

# Filter recent seasons and valid durations
pit_recent = pit_merged[
    (pit_merged["season"] >= 2022) & 
    (pit_merged["durationSeconds"] > 0) & 
    (pit_merged["durationSeconds"] < 60)
].copy()

# Average pit stop time by team
team_pit = (
    pit_recent.groupby("constructorName")
    .agg(
        avgDuration=("durationSeconds", "mean"),
        medianDuration=("durationSeconds", "median"),
        totalStops=("durationSeconds", "count"),
        fastestStop=("durationSeconds", "min")
    )
    .sort_values("medianDuration")
    .head(10)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(
    team_pit["constructorName"],
    team_pit["medianDuration"],
    color=[get_color(t) for t in team_pit["constructorName"]],
    edgecolor="white", linewidth=0.5
)

# Add value labels
for bar, val, fast in zip(bars, team_pit["medianDuration"], team_pit["fastestStop"]):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}s (best: {fast:.2f}s)", va="center", fontsize=10)

ax.set_xlabel("Median pit stop duration (seconds)")
ax.set_title("Pit stop performance by team (2022-2025)")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Overtaking kings — who gains the most positions?

# COMMAND ----------

# Average positions gained per race by driver (recent seasons)
overtakers = (
    driver_stats[driver_stats["season"] >= 2022]
    .groupby(["driverCode", "driverName"])
    .agg(
        avgGained=("avgPositionsGained", "mean"),
        totalRaces=("races", "sum"),
        constructorName=("constructorName", "last")
    )
    .reset_index()
    .sort_values("avgGained", ascending=False)
    .head(15)
)

fig, ax = plt.subplots(figsize=(14, 6))

colors = [get_color(t) for t in overtakers["constructorName"]]
bars = ax.bar(overtakers["driverCode"], overtakers["avgGained"], color=colors,
              edgecolor="white", linewidth=0.5)

# Add zero line
ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.3)

# Color negative bars differently
for bar, val in zip(bars, overtakers["avgGained"]):
    if val < 0:
        bar.set_alpha(0.5)

ax.set_xlabel("Driver")
ax.set_ylabel("Avg positions gained per race")
ax.set_title("Average positions gained per race (2022-2025)\nPositive = gains places, Negative = loses places")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Insight:** Drivers who regularly qualify at the front (e.g., Verstappen) often 
# MAGIC show negative values — they start P1 and can only lose places. Midfield 
# MAGIC drivers with strong race pace show positive values.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Teammate head-to-head battles

# COMMAND ----------

# Most recent season
latest_season = h2h["season"].max()
h2h_latest = h2h[h2h["season"] == latest_season].copy()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Race head-to-head
ax1 = axes[0]
teams = h2h_latest["constructorName"].tolist()
d1_wins = h2h_latest["driver1RaceWins"].tolist()
d2_wins = h2h_latest["driver2RaceWins"].tolist()
labels = [f"{row['driver1Code']} vs {row['driver2Code']}" for _, row in h2h_latest.iterrows()]

y_pos = range(len(teams))
ax1.barh(y_pos, d1_wins, color="#3671C6", alpha=0.8, label="Driver 1")
ax1.barh(y_pos, [-w for w in d2_wins], color="#F91536", alpha=0.8, label="Driver 2")
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=10)
ax1.set_xlabel("Race wins")
ax1.set_title(f"{latest_season} Teammate race battles")
ax1.legend()
ax1.axvline(x=0, color="black", linewidth=0.8)

# Qualifying head-to-head
ax2 = axes[1]
d1_qual = h2h_latest["driver1QualWins"].tolist()
d2_qual = h2h_latest["driver2QualWins"].tolist()

ax2.barh(y_pos, d1_qual, color="#3671C6", alpha=0.8, label="Driver 1")
ax2.barh(y_pos, [-w for w in d2_qual], color="#F91536", alpha=0.8, label="Driver 2")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=10)
ax2.set_xlabel("Qualifying wins")
ax2.set_title(f"{latest_season} Teammate qualifying battles")
ax2.legend()
ax2.axvline(x=0, color="black", linewidth=0.8)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Circuit specialists — team dominance by track

# COMMAND ----------

# Heatmap: team performance across circuits
top_circuits = (
    circuit.groupby("circuitName")["raceEntries"]
    .sum().nlargest(15).index.tolist()
)
top_constructors = ["Red Bull", "Mercedes", "Ferrari", "McLaren", "Aston Martin"]

heatmap_data = (
    circuit[
        (circuit["circuitName"].isin(top_circuits)) &
        (circuit["constructorName"].isin(top_constructors))
    ]
    .pivot_table(
        index="circuitName", columns="constructorName",
        values="avgFinish", aggfunc="mean"
    )
    .reindex(columns=top_constructors)
)

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(heatmap_data.values, cmap="RdYlGn_r", aspect="auto")

ax.set_xticks(range(len(top_constructors)))
ax.set_xticklabels(top_constructors, rotation=45, ha="right")
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)

# Add values in cells
for i in range(len(heatmap_data.index)):
    for j in range(len(top_constructors)):
        val = heatmap_data.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, color="white" if val > 8 else "black")

ax.set_title("Average finish position by circuit and team\n(lower = better, green = stronger)")
plt.colorbar(im, ax=ax, label="Avg finish position", shrink=0.8)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Strategy analysis — which pit strategy wins?

# COMMAND ----------

# Aggregate strategy outcomes
strat_summary = (
    strategy.groupby("strategyType")
    .agg(
        races=("driversUsing", "sum"),
        avgPosition=("avgFinishPosition", "mean"),
        avgPoints=("avgPoints", "mean"),
        totalWins=("wins", "sum"),
        totalPodiums=("podiums", "sum")
    )
    .reset_index()
    .sort_values("avgPosition")
)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Average finish position by strategy
ax1 = axes[0]
colors = ["#3671C6", "#F58020", "#F91536", "#888888"]
ax1.bar(strat_summary["strategyType"], strat_summary["avgPosition"],
        color=colors[:len(strat_summary)], edgecolor="white")
ax1.set_ylabel("Avg finish position")
ax1.set_title("Avg finish by strategy")
ax1.invert_yaxis()

# Wins by strategy
ax2 = axes[1]
ax2.bar(strat_summary["strategyType"], strat_summary["totalWins"],
        color=colors[:len(strat_summary)], edgecolor="white")
ax2.set_ylabel("Total wins")
ax2.set_title("Race wins by strategy")

# Usage frequency
ax3 = axes[2]
ax3.bar(strat_summary["strategyType"], strat_summary["races"],
        color=colors[:len(strat_summary)], edgecolor="white")
ax3.set_ylabel("Total driver-races")
ax3.set_title("Strategy usage frequency")

plt.suptitle("Pit stop strategy analysis (2018-2025)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Season-over-season driver improvement

# COMMAND ----------

# Track drivers who competed in multiple seasons
multi_season = (
    driver_stats.groupby("driverCode")["season"]
    .nunique().reset_index()
    .query("season >= 4")
)

interesting_drivers = ["VER", "HAM", "LEC", "NOR", "SAI", "RUS", "PER"]
multi_data = driver_stats[driver_stats["driverCode"].isin(interesting_drivers)]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Points per season
ax1 = axes[0]
for driver in interesting_drivers:
    d = multi_data[multi_data["driverCode"] == driver].sort_values("season")
    team = d["constructorName"].iloc[-1]
    ax1.plot(d["season"], d["totalPoints"], marker="o", linewidth=2,
             color=get_color(team), label=driver, markersize=5)

ax1.set_xlabel("Season")
ax1.set_ylabel("Total points")
ax1.set_title("Points trajectory by driver")
ax1.legend(ncol=2)
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Win rate per season
ax2 = axes[1]
for driver in interesting_drivers:
    d = multi_data[multi_data["driverCode"] == driver].sort_values("season")
    team = d["constructorName"].iloc[-1]
    ax2.plot(d["season"], d["winRate"], marker="o", linewidth=2,
             color=get_color(team), label=driver, markersize=5)

ax2.set_xlabel("Season")
ax2.set_ylabel("Win rate (%)")
ax2.set_title("Win rate trajectory by driver")
ax2.legend(ncol=2)
ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Reliability analysis — who finishes races?

# COMMAND ----------

# Finish rate by constructor across seasons
fig, ax = plt.subplots(figsize=(14, 6))

for team in top_constructors:
    team_data = (
        constructor_stats[constructor_stats["constructorName"] == team]
        .sort_values("season")
    )
    ax.plot(
        team_data["season"], team_data["reliabilityRate"],
        marker="s", linewidth=2, markersize=6,
        color=get_color(team), label=team
    )

ax.set_xlabel("Season")
ax.set_ylabel("Finish rate (%)")
ax.set_title("Team reliability (finish rate) by season")
ax.legend(framealpha=0.9)
ax.set_ylim(50, 105)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Points distribution — how competitive is each season?

# COMMAND ----------

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

seasons = sorted(driver_stats["season"].unique())

for i, season in enumerate(seasons):
    if i >= 8:
        break
    ax = axes[i]
    season_data = (
        driver_stats[driver_stats["season"] == season]
        .sort_values("totalPoints", ascending=False)
        .head(10)
    )
    colors = [get_color(t) for t in season_data["constructorName"]]
    ax.barh(season_data["driverCode"], season_data["totalPoints"],
            color=colors, edgecolor="white", linewidth=0.3)
    ax.set_title(str(season), fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Points" if i >= 4 else "")

# Remove extra subplots if needed
for j in range(i + 1, 8):
    axes[j].set_visible(False)

plt.suptitle("Top 10 drivers by points — season comparison", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key findings summary

# COMMAND ----------

# Auto-generate key stats
latest = driver_stats["season"].max()
champion = driver_stats[driver_stats["season"] == latest].nlargest(1, "totalPoints").iloc[0]
best_team = constructor_stats[constructor_stats["season"] == latest].nlargest(1, "totalPoints").iloc[0]

print("=" * 60)
print(f"  KEY FINDINGS ({min(seasons)}-{max(seasons)})")
print("=" * 60)
print(f"\n  {latest} Champion: {champion['driverName']} ({champion['constructorName']})")
print(f"  Points: {champion['totalPoints']}, Wins: {int(champion['wins'])}, Podiums: {int(champion['podiums'])}")
print(f"\n  {latest} Best Team: {best_team['constructorName']}")
print(f"  Points: {best_team['totalPoints']}, Wins: {int(best_team['wins'])}")

# Most wins overall
most_wins = driver_stats.groupby("driverName")["wins"].sum().nlargest(5)
print(f"\n  Most race wins ({min(seasons)}-{max(seasons)}):")
for driver, wins in most_wins.items():
    print(f"    {driver}: {int(wins)} wins")

# Best average finish
best_avg = (
    driver_stats[driver_stats["races"] >= 15]
    .groupby("driverName")["avgFinishPosition"]
    .mean().nsmallest(5)
)
print(f"\n  Best avg finish position (min 15 races):")
for driver, pos in best_avg.items():
    print(f"    {driver}: {pos:.2f}")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Analysis Complete!
# MAGIC 
# MAGIC **Visualizations created:**
# MAGIC 1. Championship points progression (line chart)
# MAGIC 2. Constructor power shifts across seasons
# MAGIC 3. Qualifying vs race performance (scatter/bubble)
# MAGIC 4. Pit stop performance by team (bar chart)
# MAGIC 5. Overtaking analysis (bar chart)
# MAGIC 6. Teammate head-to-head battles (butterfly chart)
# MAGIC 7. Circuit specialist heatmap
# MAGIC 8. Pit strategy analysis (multi-chart)
# MAGIC 9. Driver improvement trajectories
# MAGIC 10. Team reliability trends
# MAGIC 11. Season-by-season points distribution
# MAGIC 
# MAGIC **For deeper investigation:**
# MAGIC - Change the `SEASON` variable in section 1 to explore different years
# MAGIC - Modify `top_teams` or `interesting_drivers` lists to focus on specific teams/drivers
# MAGIC - Add weather data (via FastF1) to analyze wet vs dry performance
# MAGIC - Analyze safety car impact on race outcomes
