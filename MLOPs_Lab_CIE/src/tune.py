import pandas as pd
import numpy as np
import os
import json
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Create results folder
os.makedirs("results", exist_ok=True)

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("power_output_kwh", axis=1)
y = df["power_output_kwh"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Parameter grid
param_dist = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5]
}

model = GradientBoostingRegressor(random_state=42)

mlflow.set_experiment("solaredge-power-output-kwh")

with mlflow.start_run(run_name="tuning-solaredge") as parent_run:
    
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    preds = best_model.predict(X_test)

    best_mae = mean_absolute_error(y_test, preds)
    best_cv_mae = -search.best_score_

    # Log trials
    for i, params in enumerate(search.cv_results_["params"]):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_mae", -search.cv_results_["mean_test_score"][i])

output = {
    "search_type": "random",
    "n_folds": 3,
    "total_trials": 5,
    "best_params": search.best_params_,
    "best_mae": best_mae,
    "best_cv_mae": best_cv_mae,
    "parent_run_name": "tuning-solaredge"
}

with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)

print("Step 2 completed. JSON saved.")