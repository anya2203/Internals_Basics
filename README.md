
# Internals_Basics

## MLOps Lab CIE Submission

This repository contains a complete MLOps pipeline for predicting solar power output using machine learning and deployment practices.

---

## 📌 Tasks Completed

### 🔹 Task 1: Model Comparison
- Trained **Ridge** and **GradientBoosting** models
- Used **MLflow** for experiment tracking
- Evaluated models using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- Selected **Ridge** as the best model based on MAE

---

### 🔹 Task 2: Hyperparameter Tuning
- Applied **RandomizedSearchCV**
- Used **3-fold cross-validation**
- Tuned **GradientBoosting Regressor**
- Parameters used:
  - `n_estimators`
  - `learning_rate`
  - `max_depth`
- Logged tuning runs using MLflow

---

### 🔹 Task 3: Docker Packaging
- Developed CLI prediction tool (`predict_cli.py`)
- Created Docker container using:
  - Base image: `python:3.11-slim`
- Built image:
```

solaredge-predictor:v1

```
- Successfully ran container and generated predictions

---

### 🔹 Task 4: Retraining Pipeline
- Combined:
- `training_data.csv`
- `new_data.csv`
- Retrained model using updated dataset
- Compared performance with original model
- Promoted model if MAE improved

---

## 📁 Project Structure

```

MLOPs_Lab_CIE/
│── data/
│── models/
│── results/
│── src/
│── Dockerfile
│── requirements.txt

```

---

## 📊 Output Files

Located in `results/`:

- `step1_s1.json`
- `step2_s2.json`
- `step3_s3.json`
- `step4_s8.json`

---

## ⚙️ Tech Stack

- Python  
- Scikit-learn  
- MLflow  
- Docker  

---

## 👩‍💻 Author

Ananya
```



