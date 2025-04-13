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
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

app = FastAPI()

scaler = StandardScaler()
scaler.mean_ = np.array([60, 12, 2, 27, 0.5, 1500, 0.7, 1.2, 5])
scaler.scale_ = np.array([10, 3, 1, 2, 0.5, 300, 0.1, 0.2, 2])

model_mri = AlzheimerCNN()
model = torch.load("models/alzheimer_mri_model.pth", map_location=torch.device("cpu")) #cuda
model_mri.load_state_dict(model["model_state"])

class_to_idx = model["class_to_idx"]
print("Mapeamento de Classes:")
for idx, class_name in class_to_idx.items():
    print(f"{idx}: {class_name}")

model_mri.eval()

# Carregar os modelos e objetos salvos
model_csv = joblib.load('models/alzheimer_csv_model.pkl')
scaler_csv = joblib.load('models/scaler.pkl')
le = joblib.load('models/label_encoder.pkl')
selector1 = joblib.load('models/feature_selector1.pkl')
selector2 = joblib.load('models/feature_selector2.pkl')
pca = joblib.load('models/pca.pkl')

classes = ["MildImpairment", "ModerateImpairment", "NoImpairment", "VeryMildImpairment"]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Carregar o CSV com os dados
df_mmse = pd.read_csv("D:\\Projetos\\TrabalhoFinalDataScience\\AlzheimerEarlyDetection\\alzheimers_disease_data.csv")

# Função para mapear o MMSE para diagnóstico
def classify_mmse(mmse):
    if mmse >= 28:
        return "No Impairment"
    elif 24 <= mmse < 28:
        return "Very Mild Impairment"
    elif 20 <= mmse < 24:
        return "Mild Impairment"
    else:
        return "Moderate Impairment"

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
        try:
            # Criar DataFrame com os dados
            df = pd.DataFrame([record])
            
            # Remover colunas que não serão usadas
            cols_to_drop = ["PatientID", "Diagnosis", "DoctorInCharge", "MMSE"]
            features = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
            
            # Garantir que os dados são numéricos
            for col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce')
            
            # Preencher valores NaN com a média da coluna
            features = features.fillna(features.mean())
            
            # Escalonar os dados
            X_scaled = scaler_csv.transform(features)
            
            # Aplicar PCA para reduzir para 15 features
            X_pca = pca.transform(X_scaled)
            
            # Fazer a predição
            prediction = model_csv.predict(X_pca)[0]
            probabilities = model_csv.predict_proba(X_pca)[0]
            
            # Decodificar a classe predita
            predicted_class = le.inverse_transform([prediction])[0]
            
            # Criar dicionário de probabilidades
            prob_dict = {le.classes_[i]: f"{prob*100:.2f}%" for i, prob in enumerate(probabilities)}
            
            results.append({
                "PatientID": record.get("PatientID", "Unknown"),
                "PredictedDiagnosis": predicted_class,
                "Probabilities": prob_dict
            })
            
        except Exception as e:
            results.append({
                "PatientID": record.get("PatientID", "Unknown"),
                "Error": str(e)
            })
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)