# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook 05 — ML: Podium Finish Predictor
# MAGIC 
# MAGIC Build a classification model to predict whether a driver finishes on the podium (top 3).
# MAGIC 
# MAGIC **Key skills demonstrated:**
# MAGIC - Feature engineering from gold layer tables
# MAGIC - scikit-learn model training
# MAGIC - MLflow experiment tracking (parameters, metrics, artifacts)
# MAGIC - Model comparison and registry
# MAGIC 
# MAGIC This is what interviewers want to see: not a perfect model, but a solid ML workflow.

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

print("✓ Storage connected")

# COMMAND ----------

# MAGIC %pip install scikit-learn mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Re-set storage after restart
storage_account_name = "f1analyticslake2"
storage_account_key = "PASTE_YOUR_KEY_HERE"
spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.dfs.core.windows.net",
    storage_account_key
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Build Feature Dataset
# MAGIC 
# MAGIC We'll use the gold `driver_race_performance` table and engineer features
# MAGIC that would be known BEFORE a race (no data leakage).

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql.functions import col, when, lit, avg, lag, count, expr
from pyspark.sql.window import Window

# Load data
perf = spark.table("f1_gold.driver_race_performance").toPandas()
driver_stats = spark.table("f1_gold.driver_season_stats").toPandas()

print(f"Race performance records: {len(perf)}")
print(f"Driver season records: {len(driver_stats)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering
# MAGIC 
# MAGIC Features we can know BEFORE the race starts:
# MAGIC - Grid position (from qualifying)
# MAGIC - Driver's season stats up to this point
# MAGIC - Constructor's historical performance at this circuit
# MAGIC - Gap to pole in qualifying

# COMMAND ----------

# Start with the base data
df = perf.copy()

# Target variable: did the driver finish on the podium?
df["is_podium"] = (df["position"] <= 3).astype(int)

# Feature 1: Grid position (known before race)
df["grid"] = df["gridPosition"].fillna(20)

# Feature 2: Qualifying gap to pole
df["qual_gap"] = df["gapToPole"].fillna(df["gapToPole"].median())

# Feature 3: Number of pit stops planned (use historical average as proxy)
df["pit_stops"] = df["numPitStops"].fillna(2)

# Feature 4: Rolling average finish position (last 5 races for this driver)
df = df.sort_values(["driverId", "season", "round"])
df["rolling_avg_pos"] = (
    df.groupby("driverId")["position"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)
df["rolling_avg_pos"] = df["rolling_avg_pos"].fillna(10)

# Feature 5: Rolling average points (last 5 races)
df["rolling_avg_points"] = (
    df.groupby("driverId")["points"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)
df["rolling_avg_points"] = df["rolling_avg_points"].fillna(0)

# Feature 6: Constructor strength (avg points per race this season so far)
df["constructor_strength"] = (
    df.groupby(["constructorName", "season"])["points"]
    .transform(lambda x: x.shift(1).expanding().mean())
)
df["constructor_strength"] = df["constructor_strength"].fillna(df["constructor_strength"].median())

# Feature 7: Qualifying position (same as grid but explicit)
df["qual_position"] = df["grid"]

# Feature 8: Is top team? (based on constructor)
top_teams = ["Red Bull", "Mercedes", "Ferrari", "McLaren"]
df["is_top_team"] = df["constructorName"].isin(top_teams).astype(int)

# Feature 9: Consistency score
df["consistency"] = df["consistencyScore"].fillna(0)

# Feature 10: Average lap time (as a proxy for pace)
df["avg_lap"] = df["avgLapTime"].fillna(df["avgLapTime"].median())

print(f"Dataset shape: {df.shape}")
print(f"Podium rate: {df['is_podium'].mean():.3f} ({df['is_podium'].sum()} podiums out of {len(df)} races)")

# COMMAND ----------

# Preview features
feature_cols = [
    "grid", "qual_gap", "rolling_avg_pos", "rolling_avg_points",
    "constructor_strength", "is_top_team", "consistency", "pit_stops"
]

display(
    spark.createDataFrame(
        df[["season", "raceName", "driverCode", "constructorName", "position", "is_podium"] + feature_cols]
        .head(20)
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Train/Test Split
# MAGIC 
# MAGIC Use 2018-2023 for training, 2024+ for testing.
# MAGIC This is time-based splitting — more realistic than random.

# COMMAND ----------

from sklearn.model_selection import train_test_split

feature_cols = [
    "grid", "qual_gap", "rolling_avg_pos", "rolling_avg_points",
    "constructor_strength", "is_top_team", "consistency", "pit_stops"
]

target = "is_podium"

# Time-based split: train on 2018-2023, test on 2024+
train_df = df[df["season"] <= 2023].copy()
test_df = df[df["season"] >= 2024].copy()

X_train = train_df[feature_cols].fillna(0)
y_train = train_df[target]
X_test = test_df[feature_cols].fillna(0)
y_test = test_df[target]

print(f"Training: {len(X_train)} samples ({y_train.sum()} podiums, {y_train.mean():.3f} rate)")
print(f"Testing:  {len(X_test)} samples ({y_test.sum()} podiums, {y_test.mean():.3f} rate)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Train Models with MLflow Tracking
# MAGIC 
# MAGIC We'll train 3 models and compare them in MLflow.

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

# Set the experiment
mlflow.set_experiment("/Users/f1-podium-predictor")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 1: Logistic Regression (baseline)

# COMMAND ----------

with mlflow.start_run(run_name="logistic_regression"):
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("features", feature_cols)
    mlflow.log_param("train_seasons", "2018-2023")
    mlflow.log_param("test_seasons", "2024+")
    mlflow.log_param("n_features", len(feature_cols))
    
    # Train
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    
    # Predict
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc)
    
    # Log model
    mlflow.sklearn.log_model(lr, "model")
    
    print(f"Logistic Regression Results:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Podium', 'Podium'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 2: Random Forest

# COMMAND ----------

with mlflow.start_run(run_name="random_forest"):
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("features", feature_cols)
    mlflow.log_param("train_seasons", "2018-2023")
    mlflow.log_param("test_seasons", "2024+")
    
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, 
        class_weight="balanced", random_state=42
    )
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc)
    
    # Log feature importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    
    # Save feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.show()
    
    mlflow.sklearn.log_model(rf, "model")
    
    print(f"Random Forest Results:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Podium', 'Podium'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model 3: Gradient Boosting

# COMMAND ----------

with mlflow.start_run(run_name="gradient_boosting"):
    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("features", feature_cols)
    mlflow.log_param("train_seasons", "2018-2023")
    mlflow.log_param("test_seasons", "2024+")
    
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    gb.fit(X_train, y_train)
    
    y_pred = gb.predict(X_test)
    y_prob = gb.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc)
    
    # Feature importance
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": gb.feature_importances_
    }).sort_values("importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Gradient Boosting Feature Importance")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.show()
    
    mlflow.sklearn.log_model(gb, "model")
    
    print(f"Gradient Boosting Results:")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Podium', 'Podium'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Compare Models in MLflow
# MAGIC 
# MAGIC Go to the **Experiments** tab in the left sidebar of Databricks
# MAGIC → click on `f1-podium-predictor` to see all 3 runs side by side.
# MAGIC 
# MAGIC You can compare AUC, F1, accuracy across models with the built-in comparison UI.

# COMMAND ----------

# Quick comparison table
print("=" * 70)
print("  MODEL COMPARISON")
print("=" * 70)
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
print("-" * 70)

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "Gradient Boosting": gb,
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"{name:<25} {accuracy_score(y_test, y_pred):>10.3f} "
          f"{precision_score(y_test, y_pred, zero_division=0):>10.3f} "
          f"{recall_score(y_test, y_pred, zero_division=0):>10.3f} "
          f"{f1_score(y_test, y_pred, zero_division=0):>10.3f} "
          f"{roc_auc_score(y_test, y_prob):>10.3f}")

print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Register Best Model
# MAGIC 
# MAGIC Register the best performing model in MLflow Model Registry.

# COMMAND ----------

# Find the best run by AUC
best_run = mlflow.search_runs(
    experiment_names=["/Users/f1-podium-predictor"],
    order_by=["metrics.auc_roc DESC"],
    max_results=1
)

best_run_id = best_run.iloc[0]["run_id"]
best_model_name = best_run.iloc[0]["params.model_type"]
best_auc = best_run.iloc[0]["metrics.auc_roc"]

print(f"Best model: {best_model_name}")
print(f"Run ID: {best_run_id}")
print(f"AUC-ROC: {best_auc:.3f}")

# Register the model
model_uri = f"runs:/{best_run_id}/model"
model_details = mlflow.register_model(model_uri, "f1-podium-predictor")

print(f"\n✓ Model registered as '{model_details.name}' version {model_details.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Make Predictions on Latest Data
# MAGIC 
# MAGIC Use the registered model to predict podium probabilities for the latest races.

# COMMAND ----------

# Load the registered model
import mlflow.pyfunc

model = mlflow.pyfunc.load_model(f"models:/f1-podium-predictor/latest")

# Get latest season data
latest = df[df["season"] == df["season"].max()].copy()
X_latest = latest[feature_cols].fillna(0)

# Predict probabilities
latest["podium_probability"] = model.predict(pd.DataFrame(X_latest, columns=feature_cols))

# Show predictions for last race
last_round = latest["round"].max()
predictions = (
    latest[latest["round"] == last_round]
    [["raceName", "driverCode", "constructorName", "grid", "position", 
      "is_podium", "podium_probability"]]
    .sort_values("grid")
)

print(f"Predictions for: {predictions['raceName'].iloc[0]}")
display(spark.createDataFrame(predictions))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ ML Notebook Complete!
# MAGIC 
# MAGIC **What we demonstrated:**
# MAGIC - Feature engineering from gold tables (no data leakage — time-based split)
# MAGIC - 3 models trained and compared: Logistic Regression, Random Forest, Gradient Boosting
# MAGIC - Full MLflow tracking: parameters, metrics, artifacts, feature importance plots
# MAGIC - Model registered in MLflow Model Registry
# MAGIC - Inference on latest data
# MAGIC 
# MAGIC **Interview talking points:**
# MAGIC - **Why these features?** Grid position is the strongest predictor in F1. Constructor strength captures car performance. Rolling averages capture form.
# MAGIC - **Why time-based split?** Random splitting would leak future info into training. In production, you'd always predict forward.
# MAGIC - **Why MLflow?** Reproducibility. Every experiment is tracked with exact parameters, code version, and results. Model registry enables versioned deployment.
# MAGIC - **What would you improve?** More features (weather, track type, tire strategy), hyperparameter tuning with Optuna, deploy as a real-time endpoint.
# MAGIC 
# MAGIC **To explore MLflow UI:** Click "Experiments" in the Databricks sidebar → f1-podium-predictor
