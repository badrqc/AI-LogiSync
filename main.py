from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from scipy.spatial import distance_matrix
import mlflow.sklearn

app = FastAPI(title="AI-LogiSync API")

# 1. Chargement dial l-modele mn MLflow
try:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = "models:/LogiSync_Model_Prod/latest"
    model = mlflow.sklearn.load_model(model_uri)
    print("💎 Modèle chargé avec succès depuis MLflow.")
except:
    model = None
    print("⚠ Erreur: Modèle introuvable.")

# 2. Schemas dial l-données
class OrderData(BaseModel):
    dist: float
    weight: float
    traffic: int
    weather: int

class GPSPoint(BaseModel):
    id: str
    lat: float
    lon: float

# 3. Endpoint: Prediction dial l-retard
@app.post("/predict")
def predict(data: OrderData):
    if not model:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    feat = np.array([[data.dist, data.weight, data.traffic, data.weather]])
    pred = model.predict(feat)[0]
    return {"prediction": "DELAY RISK" if pred == 1 else "ON TIME"}

# 4. Endpoint: Optimization GPS (Dakchi dial l-prof)
@app.post("/optimize")
def optimize(points: List[GPSPoint]):
    # extraction Lat/Lon
    coords = [[p.lat, p.lon] for p in points]
    ids = [p.id for p in points]
    
    # Matrice dial distance (SciPy)
    # Hna kankhdmou b GPS coordinates l-ha9i9iyin
    dist_mat = distance_matrix(coords, coords)
    
    # Nearest Neighbor Algorithm
    unvisited = list(range(1, len(coords)))
    current = 0
    path = [current]
    
    while unvisited:
        next_p = min(unvisited, key=lambda i: dist_mat[current][i])
        unvisited.remove(next_p)
        path.append(next_p)
        current = next_p
        
    return {"optimized_path": [ids[i] for i in path]}