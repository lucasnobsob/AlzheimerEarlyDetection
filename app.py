#pip install pandas torch scikit-learn joblib matplotlib opencv-python fastapi Pillow torchvision

import base64
import cv2
from fastapi import FastAPI, File, UploadFile
from typing import List
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from AlzheimerCNN import AlzheimerCNN
from ClinicalData import ClinicalData
from gradcam_utils import GradCAM
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

app = FastAPI()

scaler = StandardScaler()
scaler.mean_ = np.array([60, 12, 2, 27, 0.5, 1500, 0.7, 1.2, 5])
scaler.scale_ = np.array([10, 3, 1, 2, 0.5, 300, 0.1, 0.2, 2])

model_mri = AlzheimerCNN()
model = torch.load("alzheimer_mri_model.pth", map_location=torch.device("cpu")) #cuda
model_mri.load_state_dict(model["model_state"])

class_to_idx = model["class_to_idx"]
print("Mapeamento de Classes:")
for idx, class_name in class_to_idx.items():
    print(f"{idx}: {class_name}")

model_mri.eval()

model_csv = joblib.load('alzheimer_csv_model.joblib')
scaler_csv = joblib.load('scaler.pkl')

classes = ["MildImpairment", "ModerateImpairment", "NoImpairment", "VeryMildImpairment"]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

gradcam = GradCAM(model_mri, model_mri.conv3)

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    results = {}
    
    for file in files:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)
    
        with torch.no_grad():
            output = model_mri(image)
            predicted_class = torch.argmax(output, dim=1).item()
            probabilities = F.softmax(output, dim=1)

        prob_dict = {classes[i]: f"{probabilities[0, i].item() * 100:.2f}%" for i in range(len(classes))}
        results[file.filename] = {"probabilidades": prob_dict, "classe_predita": classes[predicted_class]}

    return results

@app.post("/predict_clinical_data")
async def predict_clinical_data(data: List[dict]):
    results = []

    for record in data:
        df = pd.DataFrame([record])
        label_encoder = LabelEncoder()
        df["Gender"] = label_encoder.fit_transform(df["Gender"])

        features = df.drop(columns=["PatientID", "Diagnosis", "DoctorInCharge", "MMSE"])
        features = scaler_csv.transform(features)
        prediction = model_csv.predict(features)[0]

        diagnosis_mapping = {
            0: "No Impairment",
            1: "Mild Impairment",
            2: "Moderate Impairment",
            3: "Very Mild Impairment"
        }

        diagnosis = diagnosis_mapping.get(prediction, "Unknown")
        results.append({"PatientID": record["PatientID"], "PredictedDiagnosis": diagnosis})

    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)