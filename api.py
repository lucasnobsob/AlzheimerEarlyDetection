from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Alzheimer's Disease Prediction API",
             description="API for predicting Alzheimer's disease stages based on patient data",
             version="1.0.0")

# Verify model files exist and are readable
def verify_model_files():
    required_files = {
        'random_forest_model.pkl': 'Random Forest model',
        'scaler.pkl': 'Scaler',
        'label_encoder.pkl': 'Label Encoder'
    }
    
    for file, description in required_files.items():
        if not os.path.exists(file):
            raise FileNotFoundError(f"{description} file '{file}' not found")
        if os.path.getsize(file) == 0:
            raise ValueError(f"{description} file '{file}' is empty")
        try:
            joblib.load(file)
        except Exception as e:
            raise ValueError(f"Error loading {description} file '{file}': {str(e)}")

# Load the model and preprocessing objects
try:
    logger.info("Verifying model files...")
    verify_model_files()
    
    logger.info("Loading model files...")
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    logger.info("Model files loaded successfully")
except Exception as e:
    logger.error(f"Error loading model files: {str(e)}")
    raise

class PatientData(BaseModel):
    Age: int
    Gender: int
    Ethnicity: int
    EducationLevel: int
    BMI: float
    Smoking: int
    AlcoholConsumption: float
    PhysicalActivity: float
    DietQuality: float
    SleepQuality: float
    FamilyHistoryAlzheimers: int
    CardiovascularDisease: int
    Diabetes: int
    Depression: int
    HeadInjury: int
    Hypertension: int
    SystolicBP: int
    DiastolicBP: int
    CholesterolTotal: float
    CholesterolLDL: float
    CholesterolHDL: float
    CholesterolTriglycerides: float
    #MMSE: float
    FunctionalAssessment: float
    MemoryComplaints: int
    BehavioralProblems: int
    ADL: float
    Confusion: int
    Disorientation: int
    PersonalityChanges: int
    DifficultyCompletingTasks: int
    Forgetfulness: int

class PredictionResponse(BaseModel):
    stage: str
    probability: float

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict_dementia(patient_data: PatientData):
    try:
        logger.info("Received prediction request")
        # Convert patient data to DataFrame
        df = pd.DataFrame([patient_data.dict()])
        
        # Scale the features
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        
        # Get the stage name
        stage_name = label_encoder.inverse_transform([prediction])[0]
        
        # Get the probability for the predicted stage
        stage_probability = probability[prediction]
        
        logger.info(f"Prediction completed: {stage_name} with probability {stage_probability}")
        return {
            "stage": stage_name,
            "probability": float(stage_probability)
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_dementia_batch(patients_data: List[PatientData]):
    try:
        logger.info(f"Received batch prediction request for {len(patients_data)} patients")
        # Convert list of patient data to DataFrame
        df = pd.DataFrame([patient.dict() for patient in patients_data])
        
        # Scale the features
        scaled_features = scaler.transform(df)
        
        # Make predictions
        predictions = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            stage_name = label_encoder.inverse_transform([pred])[0]
            stage_probability = probs[pred]
            
            results.append({
                "stage": stage_name,
                "probability": float(stage_probability)
            })
        
        logger.info(f"Batch prediction completed for {len(results)} patients")
        return results
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 