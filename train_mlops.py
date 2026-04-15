import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

db_path = os.path.abspath("mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("AI_LogiSync_Optimization")

X = np.array([
    [5, 10, 1, 1], 
    [25, 50, 4, 2], 
    [10, 5, 2, 1],
    [40, 100, 5, 3], 
    [2, 1, 1, 1], 
    [30, 80, 3, 2]
])

y = np.array([0, 1, 0, 1, 0, 1])

print("🚀 Démarrage de l'entraînement...")

with mlflow.start_run():
    params = {"n_estimators": 50, "max_depth": 5, "random_state": 42}
    model = RandomForestClassifier(**params)
    model.fit(X, y)

    acc = accuracy_score(y, model.predict(X))

    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="LogiSync_Model_Prod"
    ) 

    print(f"✅ Modèle enregistré ! Accuracy: {acc*100}%")