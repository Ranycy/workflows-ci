import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

mlflow.set_tracking_uri("https://dagshub.com/Ranycy/endometriosis-new.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

# 0. SET TRACKING URI 
mlflow.set_tracking_uri("https://dagshub.com/Ranycy/endometriosis-new.mlflow")

# 1. SET EXPERIMENT
mlflow.set_experiment("Endometriosis_CI")

# 2. LOAD DATA 
df = pd.read_csv("endoclean_preprocessing.csv")

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. START MLFLOW RUN
with mlflow.start_run(run_name="CI_RandomForest"):

    # 5. MODEL FINAL
    model = RandomForestClassifier(
        n_estimators=200,      
        max_depth=20,          
        min_samples_split=2,   
        random_state=42
    )

    # 6. TRAIN MODEL
    model.fit(X_train, y_train)

    # 7. PREDICT
    y_pred = model.predict(X_test)

    # 8. METRICS
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # 9. LOG PARAMETER 
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 20)
    mlflow.log_param("min_samples_split", 2)

    # 10. LOG METRICS
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # 11. SAVE MODEL 
    mlflow.sklearn.log_model(model, "model")

    print("=== HASIL MODEL CI ===")
    print("Accuracy:", acc)
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
