from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load('models/model.pkl')

class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}