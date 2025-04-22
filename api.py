from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List
import logging
from PatientData import PatientData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Alzheimer's Disease Prediction API",
             description="API for predicting Alzheimer's disease stages based on patient data",
             version="1.0.0")

model = joblib.load('models/RandomForest.joblib')

CLASS_MAPPING = {
    0: "Sem Demência",
    1: "Demência Muito Leve",
    2: "Demência Leve",
    3: "Demência Moderada"
}

class PredictionResponse(BaseModel):
    stage: str
    probability: float
    all_probabilities: Dict[str, float]

@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_dementia_batch(patients_data: List[PatientData]):
    try:
        logger.info(f"Received batch prediction request for {len(patients_data)} patients")
        df = pd.DataFrame([patient.dict() for patient in patients_data])
        
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            stage_name = CLASS_MAPPING.get(int(pred), "Unknown")
            stage_probability = probs[np.where(model.classes_ == pred)[0][0]]
            
            all_probs = {CLASS_MAPPING.get(int(cls), "Unknown"): float(prob) 
                        for cls, prob in zip(model.classes_, probs)}
            
            results.append({
                "diagnóstico": stage_name,
                "probabilidade": float(stage_probability),
                "todas_as_probabilidades": all_probs
            })
        
        logger.info(f"Batch prediction completed for {len(results)} patients")
        return results
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 