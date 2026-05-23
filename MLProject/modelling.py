import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("endoclean_preprocessing.csv")

X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]


# =========================
# 2. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 3. MLflow RUN
# =========================
with mlflow.start_run(run_name="manual_run"):

    mlflow.sklearn.autolog()

    # model 
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # =========================
    # 4. PREDICTION + METRICS
    # =========================
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # =========================
    # 5. LOG METRICS MANUAL
    # =========================
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # =========================
    # 6. LOG MODEL
    # =========================
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

    print(f"Accuracy: {acc}")
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")
    print(f"F1 Score: {f1}")
