from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List
import logging
import torch
import uvicorn
import json
from models.Helper import format_probabilities, preprocess_image
from models.PredictionResponse import PredictionResponse
from models.AlzheimerCNN import AlzheimerCNN
from models.PatientData import PatientData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Alzheimer's Disease Prediction API",
             description="API for predicting Alzheimer's disease stages based on patient data",
             version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model_csv = joblib.load('models/XGBoost.joblib')

# Load the model and class mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlzheimerCNN(num_classes=4)
model.load_state_dict(torch.load('models/alzheimer_mri_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Load CLASS_MAPPING from JSON
with open('models/class_mapping.json', 'r') as f:
    CLASS_MAPPING = json.load(f)

@app.get("/")
async def root():
    return {"message": "Welcome to the Alzheimer Detection API"}

@app.post("/predict/mri")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Preprocess the image
        image_tensor = preprocess_image(contents)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Prepare response
        result = {
            "predicted_class": CLASS_MAPPING[str(predicted_class)],
            "confidence": f"{float(confidence)*100:.2f}%",
            "probabilities": format_probabilities({
                class_name: prob for class_name, prob in zip(
                    CLASS_MAPPING.values(),
                    probabilities[0].cpu().numpy()
                )
            })
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/mri/batch")
async def batch_predict(files: List[UploadFile] = File(...)):
    try:
        results = []
        
        for file in files:
            # Read the uploaded file
            contents = await file.read()
            
            # Preprocess the image
            image_tensor = preprocess_image(contents)
            image_tensor = image_tensor.to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Prepare result for this image
            result = {
                "filename": file.filename,
                "predicted_class": CLASS_MAPPING[str(predicted_class)],
                "confidence": f"{float(confidence)*100:.2f}%",
                "probabilities": format_probabilities({
                    class_name: prob for class_name, prob in zip(
                        CLASS_MAPPING.values(),
                        probabilities[0].cpu().numpy()
                    )
                })
            }
            
            results.append(result)
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/clinical_data/batch", response_model=List[PredictionResponse])
async def predict_dementia_batch(patients_data: List[PatientData]):
    try:
        logger.info(f"Received batch prediction request for {len(patients_data)} patients")
        df = pd.DataFrame([patient.dict() for patient in patients_data])
        
        predictions = model_csv.predict(df)
        probabilities = model_csv.predict_proba(df)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            stage_name = CLASS_MAPPING.get(str(int(pred)), "Unknown")
            stage_probability = probs[np.where(model_csv.classes_ == pred)[0][0]]
            
            all_probs = {CLASS_MAPPING.get(str(int(cls)), "Unknown"): float(prob) 
                        for cls, prob in zip(model_csv.classes_, probs)}
            
            results.append({
                "stage": stage_name,
                "probability": float(stage_probability),
                "all_probabilities": all_probs
            })
        
        logger.info(f"Batch prediction completed for {len(results)} patients")
        return results
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)