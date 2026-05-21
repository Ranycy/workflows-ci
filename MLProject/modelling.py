#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# 1. SET EXPERIMENT
mlflow.set_experiment("Endometriosis_Model_SMOTE")

# 2. LOAD DATA
df = pd.read_csv("endoclean_preprocessing.csv")

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# 3. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 5. MODEL
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# 6. PREDICT
y_pred = model.predict(X_test)

# 7. METRICS
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 8. LOGGING (TANPA start_run!)
mlflow.log_param("model", "RandomForest")
mlflow.log_param("n_estimators", 300)
mlflow.log_param("SMOTE", True)
mlflow.log_param("class_weight", "balanced")

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("f1_score", f1)

mlflow.sklearn.log_model(model, "model")

print("Accuracy:", acc)
print("F1:", f1)
