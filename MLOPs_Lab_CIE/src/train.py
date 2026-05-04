import pandas as pd
import numpy as np
import os
import json
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create folders
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("power_output_kwh", axis=1)
y = df["power_output_kwh"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow experiment
mlflow.set_experiment("solaredge-power-output-kwh")

results = []

with mlflow.start_run(run_name="baseline_comparison"):

    # -------- Ridge --------
    with mlflow.start_run(run_name="Ridge", nested=True):
        ridge = Ridge()
        ridge.fit(X_train, y_train)
        preds = ridge.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_params({"model": "Ridge"})
        mlflow.log_metrics({"mae": mae, "rmse": rmse})
        mlflow.set_tag("experiment_type", "baseline_comparison")

        results.append({
            "name": "Ridge",
            "mae": mae,
            "rmse": rmse
        })

    # -------- GradientBoosting --------
    with mlflow.start_run(run_name="GradientBoosting", nested=True):
        gbr = GradientBoostingRegressor(random_state=42)
        gbr.fit(X_train, y_train)
        preds = gbr.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_params({"model": "GradientBoosting"})
        mlflow.log_metrics({"mae": mae, "rmse": rmse})
        mlflow.set_tag("experiment_type", "baseline_comparison")

        results.append({
            "name": "GradientBoosting",
            "mae": mae,
            "rmse": rmse
        })

# Select best model
best = min(results, key=lambda x: x["mae"])

output = {
    "experiment_name": "solaredge-power-output-kwh",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}

# Save JSON
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

# Save best model
if best["name"] == "Ridge":
    final_model = Ridge()
else:
    final_model = GradientBoostingRegressor(random_state=42)

final_model.fit(X_train, y_train)
joblib.dump(final_model, "models/model.joblib")

print("Step 1 completed. JSON + model saved.")