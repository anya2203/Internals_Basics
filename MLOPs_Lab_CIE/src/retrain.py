import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Load datasets
train_df = pd.read_csv("data/training_data.csv")
new_df = pd.read_csv("data/new_data.csv")

# Combine
combined_df = pd.concat([train_df, new_df], ignore_index=True)

# Split ORIGINAL (for fair comparison)
X_orig = train_df.drop("power_output_kwh", axis=1)
y_orig = train_df["power_output_kwh"]

X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    X_orig, y_orig, test_size=0.2, random_state=42
)

# Train champion (original)
champion = Ridge()
champion.fit(X_train_o, y_train_o)
preds_o = champion.predict(X_test_o)
champion_mae = mean_absolute_error(y_test_o, preds_o)

# Train retrained model on combined data
X_comb = combined_df.drop("power_output_kwh", axis=1)
y_comb = combined_df["power_output_kwh"]

X_train_c, _, y_train_c, _ = train_test_split(
    X_comb, y_comb, test_size=0.2, random_state=42
)

retrained = Ridge()
retrained.fit(X_train_c, y_train_c)

# Evaluate retrained on SAME original test set
preds_c = retrained.predict(X_test_o)
retrained_mae = mean_absolute_error(y_test_o, preds_c)

# Compare
improvement = champion_mae - retrained_mae

action = "promoted" if improvement > 0 else "kept_champion"

output = {
    "original_data_rows": len(train_df),
    "new_data_rows": len(new_df),
    "combined_data_rows": len(combined_df),
    "champion_mae": champion_mae,
    "retrained_mae": retrained_mae,
    "improvement": improvement,
    "min_improvement_threshold": 0,
    "action": action,
    "comparison_metric": "mae"
}

os.makedirs("results", exist_ok=True)

with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("Step 4 completed. JSON saved.")