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
from fastapi.responses import JSONResponse
from models.Helper import format_probabilities, preprocess_image
from models.PredictionResponse import PredictionResponse
from models.AlzheimerCNN import AlzheimerCNN
from models.PatientData import PatientData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Alzheimer's Disease Prediction API",
    description="API for predicting Alzheimer's disease stages using MRI scans and clinical patient data. Supports single and batch predictions for MRI images and clinical data analysis.",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Health Check",
            "description": "Endpoints for checking the API's health and status."
        },
        {
            "name": "MRI Predictions",
            "description": "Endpoints for predicting Alzheimer's disease stages based on MRI scan images."
        },
        {
            "name": "Clinical Data Predictions",
            "description": "Endpoints for predicting Alzheimer's disease stages based on clinical patient data."
        }
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model_csv = joblib.load('models/RandomForest.joblib')

# Load the model and class mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlzheimerCNN(num_classes=4)
model.load_state_dict(torch.load('models/alzheimer_mri_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Load CLASS_MAPPING from JSON
with open('models/class_mapping.json', 'r') as f:
    CLASS_MAPPING = json.load(f)

class MRIResponse(BaseModel):
    predicted_class: str
    confidence: str
    probabilities: dict

class BatchMRIResponse(BaseModel):
    filename: str
    predicted_class: str
    confidence: str
    probabilities: dict

@app.get(
    "/",
    tags=["Health Check"],
    summary="API Root Endpoint",
    description="Returns a welcome message to confirm the API is running.",
    response_description="A JSON object containing a welcome message."
)
async def root():
    return {"message": "Welcome to the Alzheimer Detection API"}

@app.post(
    "/predict/mri",
    tags=["MRI Predictions"],
    summary="Predict Alzheimer's Stage from MRI Scan",
    description="Upload a single MRI scan image to predict the Alzheimer's disease stage. The response includes the predicted class, confidence score, and probability distribution across all classes.",
    response_description="A JSON object containing the predicted class, confidence, and probabilities.",
    response_model=MRIResponse
)
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
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/predict/mri/batch",
    tags=["MRI Predictions"],
    summary="Batch Predict Alzheimer's Stages from MRI Scans",
    description="Upload multiple MRI scan images to predict Alzheimer's disease stages for each. The response includes predictions for each image, including filename, predicted class, confidence, and probabilities.",
    response_description="A JSON object containing a list of prediction results for each uploaded MRI image.",
    response_model=List[BatchMRIResponse]
)
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
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/predict/clinical_data/batch",
    tags=["Clinical Data Predictions"],
    summary="Batch Predict Alzheimer's Stages from Clinical Data",
    description="Submit clinical data for multiple patients to predict Alzheimer's disease stages. The response includes the predicted stage, probability, and probability distribution for each patient.",
    response_description="A list of JSON objects containing the predicted stage, probability, and all probabilities for each patient.",
    response_model=List[PredictionResponse]
)
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
            
            formatted_probs = format_probabilities(all_probs)
            formatted_stage_probability = f"{float(stage_probability)*100:.2f}%"

            results.append({
                "stage": stage_name,
                "probability": formatted_stage_probability,
                "all_probabilities": formatted_probs
            })
        
        logger.info(f"Batch prediction completed for {len(results)} patients")
        return results
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)