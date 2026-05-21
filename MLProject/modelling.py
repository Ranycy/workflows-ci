#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_experiment("Endometriosis_CI")

df = pd.read_csv("endoclean_preprocessing.csv")

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# logging (TANPA start_run)
mlflow.log_param("model", "RandomForest")
mlflow.log_metric("accuracy", acc)
mlflow.log_metric("f1_score", f1)

mlflow.sklearn.log_model(model, "model")

print(acc, f1)
