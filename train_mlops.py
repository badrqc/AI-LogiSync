import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# 1. Configuration dial MLflow
# Kansajlo l-m'loumat f base sqlite bach n-choufouhom f l-interface
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("AI_LogiSync_Optimization")
# 2. Preparation dial l-m'loumat (Simulation)
# X: [Distance(km), Poids(kg), Trafic(1-5), Météo(1-3)]
X = np.array([[5, 10, 1, 1], [25, 50, 4, 2], [10, 5, 2, 1],
[40, 100, 5, 3], [2, 1, 1, 1], [30, 80, 3, 2]])
# y: 0 (On Time), 1 (Delayed)
y = np.array([0, 1, 0, 1, 0, 1])
print("🚀 Démarrage de l'entraînement...")
with mlflow.start_run():
    # Model: Random Forest (Algorithme s7i7)
    params = {"n_estimators": 50, "max_depth": 5, "random_state": 42}
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    # Calcul dial l-accuracy
    acc = accuracy_score(y, model.predict(X))
    # MLOPS Tracking: Kansajlo l-params w l-metrics
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    # Model Registry: Kansajlo l-modele bach l-API tchargih mn b'd
    mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="LogiSync_Model_Prod"
    ) 
    print(f"✅ Modèle enregistré ! Accuracy: {acc*100}%")