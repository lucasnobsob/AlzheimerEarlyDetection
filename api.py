from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import uvicorn
from AlzheimerCNN import AlzheimerCNN
from PatientData import PatientData

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

model_csv = joblib.load('models/RandomForest.joblib')

# Define the same transforms used in training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlzheimerCNN(num_classes=4)
#model.load_state_dict(torch.load('alzheimer_mri_model.pth', map_location=device))
model.load_state_dict(torch.load('models/best_model_resolution_invariant.pth', map_location=device))
model = model.to(device)
model.eval()

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
        
        predictions = model_csv.predict(df)
        probabilities = model_csv.predict_proba(df)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            stage_name = CLASS_MAPPING.get(int(pred), "Unknown")
            stage_probability = probs[np.where(model_csv.classes_ == pred)[0][0]]
            
            all_probs = {CLASS_MAPPING.get(int(cls), "Unknown"): float(prob) 
                        for cls, prob in zip(model_csv.classes_, probs)}
            
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

def preprocess_image(image_bytes):
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply transforms
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def format_probabilities(probabilities):
    """Format probabilities as percentages with 2 decimal places"""
    return {class_name: f"{float(prob)*100:.2f}%" for class_name, prob in probabilities.items()}

@app.get("/")
async def root():
    return {"message": "Welcome to the Alzheimer Detection API"}

@app.post("/predict")
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
            "predicted_class": CLASS_MAPPING[predicted_class],
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

@app.post("/batch_predict")
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
                "predicted_class": CLASS_MAPPING[predicted_class],
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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 