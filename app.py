from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import pandas as pd
import joblib  # Para carregar o modelo clínico
from sklearn.preprocessing import StandardScaler
from AlzheimerCNN import AlzheimerCNN
from ClinicalData import ClinicalData
from sklearn.ensemble import RandomForestClassifier  # Modelo para dados clínicos
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Padronizador dos dados (deve ser o mesmo usado no treino)
scaler = StandardScaler()
scaler.mean_ = np.array([60, 12, 2, 27, 0.5, 1500, 0.7, 1.2, 5])  # Substitua pelos valores do dataset
scaler.scale_ = np.array([10, 3, 1, 2, 0.5, 300, 0.1, 0.2, 2])

# modelo somente com as imagens
model_mri = AlzheimerCNN()
model_mri.load_state_dict(torch.load("alzheimer_mri_model.pth", map_location=torch.device("cpu")))  # Se precisar rodar na GPU, mude para 'cuda'
model_mri.eval()  # Coloca o modelo em modo de avaliação

# Carregar o modelo para os dados clínicos (CSV)
model_csv = joblib.load('alzheimer_csv_model.joblib')
# Carregar o modelo treinado e o scaler
scaler_csv = joblib.load('scaler.pkl')

# Definição das classes (substitua pelos seus nomes reais)
classes = ["MildDemented", "ModerateDemented", "VeryMildDemented", "NonDemented"]

# Transformação da imagem para o modelo (ajuste conforme seu treino)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Tamanho usado no treinamento
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalização usada no treino
])

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    results = {}
    
    for file in files:
        # Lê a imagem
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")  # Converte para RGB
        image = transform(image).unsqueeze(0)  # Aplica transformações e adiciona dimensão de batch
    
        # Faz a predição
        with torch.no_grad():
            output = model_mri(image)
            predicted_class = torch.argmax(output, dim=1).item()

        results[file.filename] = {"classe_predita": classes[predicted_class]}

    return results

@app.post("/predict_clinical_data")
async def predict_clinical_data(data: List[dict]):
    results = []

    for record in data:
        # Converter os dados clínicos para um DataFrame
        df = pd.DataFrame([record])

        # Pré-processamento para dados clínicos
        # Codificar variáveis categóricas (como Gender, Ethnicity, etc.)
        label_encoder = LabelEncoder()
        df["Gender"] = label_encoder.fit_transform(df["Gender"])

        # Normalizar os dados
        features = df.drop(columns=["PatientID", "Diagnosis", "DoctorInCharge"])
        features = scaler_csv.transform(features)

        # Fazer a predição com o modelo de dados clínicos
        prediction = model_csv.predict(features)

        diagnosis_mapping = {
            0: "No Impairment",
            1: "Mild Impairment",
            2: "Moderate Impairment",
            3: "Very Mild Impairment"
        }

        diagnosis = diagnosis_mapping.get(prediction, "Unknown")
        
        results.append({"PatientID": record["PatientID"], "PredictedDiagnosis": diagnosis})

    return results


# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)