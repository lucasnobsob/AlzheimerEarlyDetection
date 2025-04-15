from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import uvicorn
from typing import List
import os

# Import the model class
from AlzheimerCNN import AlzheimerCNN

app = FastAPI(title="Alzheimer Detection API",
             description="API for detecting Alzheimer's disease from MRI images",
             version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the same transforms used in training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class mapping
class_mapping = {
    0: "Mild Impairment",
    1: "Moderate Impairment",
    2: "No Impairment",
    3: "Very Mild Impairment"
}

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlzheimerCNN(num_classes=4)
model.load_state_dict(torch.load('best_model_resolution_invariant.pth', map_location=device))
model = model.to(device)
model.eval()

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
            "predicted_class": class_mapping[predicted_class],
            "confidence": f"{float(confidence)*100:.2f}%",
            "probabilities": format_probabilities({
                class_name: prob for class_name, prob in zip(
                    class_mapping.values(),
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
                "predicted_class": class_mapping[predicted_class],
                "confidence": f"{float(confidence)*100:.2f}%",
                "probabilities": format_probabilities({
                    class_name: prob for class_name, prob in zip(
                        class_mapping.values(),
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