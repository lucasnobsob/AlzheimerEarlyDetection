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
model = torch.load("alzheimer_mri_model.pth", map_location=torch.device("cpu")) #cuda
model_mri.load_state_dict(model["model_state"])

class_to_idx = model["class_to_idx"]
print("Mapeamento de Classes:")
for idx, class_name in class_to_idx.items():
    print(f"{idx}: {class_name}")

model_mri.eval()

model_csv = joblib.load('cognitive_impairment_model.pkl')
scaler_csv = joblib.load('scaler.pkl')

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
    total = 0
    acertos = 0

    for record in data:
        df = pd.DataFrame([record])
        label_encoder = LabelEncoder()
        df["Gender"] = label_encoder.fit_transform(df["Gender"])

        # Codificar todas as colunas categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ["PatientID", "Diagnosis", "DoctorInCharge", "MMSE"]:  # Exclua colunas não usadas
                df[col] = LabelEncoder().fit_transform(df[col])
                
        cols_to_drop = ["PatientID", "Diagnosis", "DoctorInCharge", "MMSE"]
        features = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        # Escalonar
        features = scaler_csv.transform(features)
        prediction = model_csv.predict(features)[0]

        diagnosis_mapping = {
            0: "No Impairment",
            1: "Mild Impairment",
            2: "Moderate Impairment",
            3: "Very Mild Impairment"
        }

        predicted_diagnosis = diagnosis_mapping.get(prediction, "Unknown")

        # Recuperar MMSE original pelo PatientID
        try:
            patient_id = int(record["PatientID"])
        except (ValueError, TypeError):
            patient_id = record["PatientID"]

        patient_row = df_mmse[df_mmse["PatientID"] == patient_id]

        if not patient_row.empty:
            mmse_value = patient_row.iloc[0]["MMSE"]
            original_mmse_diagnosis = classify_mmse(mmse_value)
        else:
            mmse_value = None
            original_mmse_diagnosis = "Unknown"

        # Contabiliza se houve acerto
        if predicted_diagnosis == original_mmse_diagnosis and predicted_diagnosis != "Unknown":
            acertos += 1
        total += 1

        results.append({
            "PatientID": patient_id,
            "PredictedDiagnosis": predicted_diagnosis,
            "OriginalMMSE": original_mmse_diagnosis
        })

    # Calcula acurácia
    accuracy_percent = round((acertos / total) * 100, 2) if total > 0 else 0.0

    return {
        "results": results,
        "accuracy": f"{accuracy_percent}%"
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)