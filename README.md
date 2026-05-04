# \# Internals\_Basics

# 

# \## MLOps Lab CIE Submission

# 

# This repository contains the implementation of a complete MLOps pipeline for predicting solar power output.

# 

# \---

# 

# \## 📌 Tasks Completed

# 

# \### 🔹 Task 1 — Experiment Tracking \& Model Comparison

# \- Trained \*\*Ridge\*\* and \*\*GradientBoosting\*\* models

# \- Used \*\*MLflow\*\* for experiment tracking

# \- Evaluated using:

# &#x20; - MAE

# &#x20; - RMSE

# \- Selected best model based on MAE

# 

# \---

# 

# \### 🔹 Task 2 — Hyperparameter Tuning

# \- Used \*\*RandomizedSearchCV\*\*

# \- 3-fold cross-validation

# \- Tuned \*\*GradientBoosting\*\* using:

# &#x20; - n\_estimators

# &#x20; - learning\_rate

# &#x20; - max\_depth

# \- Logged runs in MLflow

# 

# \---

# 

# \### 🔹 Task 3 — Docker Packaging

# \- Created CLI prediction tool (`predict\_cli.py`)

# \- Built Docker image:

# &#x20; solaredge-predictor:v1

# \- Tested prediction via Docker container

# 

# \---

# 

# \### 🔹 Task 4 — Retraining Pipeline

# \- Combined training + new data

# \- Retrained model

# \- Compared MAE with champion model

# \- Promoted model if improvement observed

# 

# \---

# 

# \## 📁 Project Structure

# 

# MLOPs\_Lab\_CIE/

# │── data/

# │── models/

# │── results/

# │── src/

# │── Dockerfile

# │── requirements.txt

# 

# \---

# 

# \## 📊 Output Files

# 

# Located in `results/`:

# \- step1\_s1.json

# \- step2\_s2.json

# \- step3\_s3.json

# \- step4\_s8.json

# 

# \---

# 

# \## ⚙️ Tech Stack

# \- Python

# \- Scikit-learn

# \- MLflow

# \- Docker

# 

# \---

# 

# \## 👩‍💻 Author

# Ananya

