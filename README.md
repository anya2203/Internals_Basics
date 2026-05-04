# Internals\_Basics

# 

# MLOps Lab CIE Submission

# 

# This repository contains a complete MLOps pipeline for solar power prediction.

# 

# \--------------------------------------------------

# 

# TASK 1: Model Comparison

# \- Trained Ridge and GradientBoosting models

# \- Used MLflow for experiment tracking

# \- Evaluated using MAE and RMSE

# \- Selected best model based on MAE (Ridge)

# 

# \--------------------------------------------------

# 

# TASK 2: Hyperparameter Tuning

# \- Used RandomizedSearchCV

# \- 3-fold cross-validation

# \- Tuned GradientBoosting model

# \- Parameters used:

# &#x20; n\_estimators, learning\_rate, max\_depth

# 

# \--------------------------------------------------

# 

# TASK 3: Docker

# \- Created CLI prediction tool (predict\_cli.py)

# \- Built Docker image: solaredge-predictor:v1

# \- Ran container and generated predictions

# 

# \--------------------------------------------------

# 

# TASK 4: Retraining

# \- Combined training\_data.csv and new\_data.csv

# \- Retrained model

# \- Compared MAE with original model

# \- Promoted model if improvement observed

# 

# \--------------------------------------------------

# 

# PROJECT STRUCTURE

# 

# MLOPs\_Lab\_CIE/

# &#x20; data/

# &#x20; models/

# &#x20; results/

# &#x20; src/

# &#x20; Dockerfile

# &#x20; requirements.txt

# 

# \--------------------------------------------------

# 

# OUTPUT FILES (results/)

# \- step1\_s1.json

# \- step2\_s2.json

# \- step3\_s3.json

# \- step4\_s8.json

# 

# \--------------------------------------------------

# 

# TECH STACK

# Python, Scikit-learn, MLflow, Docker

# 

# \--------------------------------------------------

# 

# AUTHOR

# Ananya

