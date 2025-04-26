import numpy as np
import pandas as pd
from scipy.stats import truncnorm

np.random.seed(42)
total_samples = 10000

data = {
    'PatientID': np.arange(4751, 4751 + total_samples, dtype=np.int64),
    'Age': np.zeros(total_samples, dtype=np.int64),
    'Gender': np.zeros(total_samples, dtype=np.int64),
    'Ethnicity': np.zeros(total_samples, dtype=np.int64),
    'EducationLevel': np.zeros(total_samples, dtype=np.int64),
    'BMI': np.zeros(total_samples, dtype=np.float64),
    'Smoking': np.zeros(total_samples, dtype=np.int64),
    'AlcoholConsumption': np.zeros(total_samples, dtype=np.float64),
    'PhysicalActivity': np.zeros(total_samples, dtype=np.float64),
    'DietQuality': np.zeros(total_samples, dtype=np.float64),
    'SleepQuality': np.zeros(total_samples, dtype=np.float64),
    'FamilyHistoryAlzheimers': np.zeros(total_samples, dtype=np.int64),
    'CardiovascularDisease': np.zeros(total_samples, dtype=np.int64),
    'Diabetes': np.zeros(total_samples, dtype=np.int64),
    'Depression': np.zeros(total_samples, dtype=np.int64),
    'HeadInjury': np.zeros(total_samples, dtype=np.int64),
    'Hypertension': np.zeros(total_samples, dtype=np.int64),
    'SystolicBP': np.zeros(total_samples, dtype=np.float64),
    'DiastolicBP': np.zeros(total_samples, dtype=np.float64),
    'CholesterolTotal': np.zeros(total_samples, dtype=np.float64),
    'CholesterolLDL': np.zeros(total_samples, dtype=np.float64),
    'CholesterolHDL': np.zeros(total_samples, dtype=np.float64),
    'CholesterolTriglycerides': np.zeros(total_samples, dtype=np.float64),
    'MMSE': np.zeros(total_samples, dtype=np.float64),
    'FunctionalAssessment': np.zeros(total_samples, dtype=np.float64),
    'MemoryComplaints': np.zeros(total_samples, dtype=np.int64),
    'BehavioralProblems': np.zeros(total_samples, dtype=np.int64),
    'ADL': np.zeros(total_samples, dtype=np.float64),
    'Confusion': np.zeros(total_samples, dtype=np.int64),
    'Disorientation': np.zeros(total_samples, dtype=np.int64),
    'PersonalityChanges': np.zeros(total_samples, dtype=np.int64),
    'DifficultyCompletingTasks': np.zeros(total_samples, dtype=np.int64),
    'Forgetfulness': np.zeros(total_samples, dtype=np.int64),
    'Diagnosis': np.zeros(total_samples, dtype=np.int64),
    'DoctorInCharge': np.array(['XXXConfid'] * total_samples, dtype=object)
}

mmse_ranges = {
    'No Impairment': (27, 29.99),
    'Very Mild': (24, 26.99),
    'Mild': (19, 23.99),
    'Moderate': (0, 18.99)
}

n_samples_per_category = {
    'No Impairment': int(total_samples * 0.50),
    'Very Mild': int(total_samples * 0.20),
    'Mild': int(total_samples * 0.15),
    'Moderate': int(total_samples * 0.15)
}

mmse_values = []
for category, (min_val, max_val) in mmse_ranges.items():
    n_samples = n_samples_per_category[category]
    values = np.random.uniform(min_val, max_val, n_samples)
    mmse_values.extend(values)
np.random.shuffle(mmse_values)
data['MMSE'] = np.clip(np.array(mmse_values), 0, 30)

def correlated_binary(prob_base, mmse, strength, inverse=False):
    scaled_mmse = (mmse - 0) / (29.99 - 0)
    if inverse:
        scaled_mmse = 1 - scaled_mmse
    strength_arr = np.full_like(scaled_mmse, strength, dtype=float)
    strength_arr = np.where(mmse < 19, np.clip(strength_arr + 0.2, 0, 1), strength_arr)
    prob = prob_base + strength_arr * (scaled_mmse - 0.5)
    prob = np.clip(prob, 0.1, 0.9)
    return np.random.binomial(1, prob)

def correlated_continuous(mmse, min_val, max_val, strength, inverse=False, dist='uniform'):
    scaled_mmse = (mmse - 0) / (29.99 - 0)
    if inverse:
        scaled_mmse = 1 - scaled_mmse
    strength_arr = np.full_like(scaled_mmse, strength, dtype=float)
    strength_arr = np.where(mmse < 19, np.clip(strength_arr + 0.2, 0, 1), strength_arr)
    mean = min_val + (max_val - min_val) * (scaled_mmse * strength_arr + (1 - strength_arr) * 0.5)
    if dist == 'normal':
        std = (max_val - min_val) / 8
        values = truncnorm.rvs((min_val - mean) / std, (max_val - mean) / std,
                               loc=mean, scale=std, size=len(mean))
    else:
        rand_unif = np.random.uniform(min_val, max_val, len(mean))
        values = rand_unif * (1 - strength_arr) + mean * strength_arr
    return np.clip(values, min_val, max_val)

def colesterol_realista(mmse, age, min_val, max_val):
    age_scaled = (age - 50) / (90 - 50)
    mmse_scaled = (29.99 - mmse) / 29.99
    combined_effect = 0.2 * mmse_scaled + 0.1 * age_scaled + 0.7 * 0.5
    mean = min_val + (max_val - min_val) * combined_effect
    values = np.random.uniform(mean - 15, mean + 15, len(mean))
    return np.clip(values, min_val, max_val)

# Grupo 1
data['Age'] = np.clip(correlated_continuous(data['MMSE'], 50, 90, 0.7, inverse=True, dist='normal').astype(np.int64), 50, 90)
data['FamilyHistoryAlzheimers'] = correlated_binary(0.5, data['MMSE'], 0.7, inverse=True)

education_levels = np.zeros(total_samples, dtype=np.int64)
for i in range(total_samples):
    mmse_score = data['MMSE'][i]
    if mmse_score >= 27:
        education_levels[i] = np.random.choice([0,1,2,3,4], p=[0.05,0.05,0.05,0.45,0.40])
    elif mmse_score >= 24:
        education_levels[i] = np.random.choice([0,1,2,3,4], p=[0.15,0.20,0.20,0.25,0.20])
    elif mmse_score >= 19:
        education_levels[i] = np.random.choice([0,1,2,3,4], p=[0.25,0.25,0.25,0.15,0.10])
    else:
        education_levels[i] = np.random.choice([0,1,2,3,4], p=[0.35,0.30,0.25,0.05,0.05])
data['EducationLevel'] = education_levels

data['FunctionalAssessment'] = correlated_continuous(data['MMSE'], 0, 9.99, 0.9)
data['MemoryComplaints'] = correlated_binary(0.5, data['MMSE'], 0.7, inverse=True)

# Grupo 2
data['CardiovascularDisease'] = correlated_binary(0.4, data['MMSE'], 0.4, inverse=True)
data['Hypertension'] = correlated_binary(0.5, data['MMSE'], 0.4, inverse=True)
data['SystolicBP'] = np.clip(correlated_continuous(data['MMSE'], 90, 180, 0.6, inverse=True, dist='normal').astype(np.int64), 90, 180)
data['DiastolicBP'] = np.clip(correlated_continuous(data['MMSE'], 60, 110, 0.6, inverse=True, dist='normal').astype(np.int64), 60, 110)
data['CholesterolTotal'] = colesterol_realista(data['MMSE'], data['Age'], 150, 300)
data['CholesterolLDL'] = colesterol_realista(data['MMSE'], data['Age'], 70, 200)
data['CholesterolTriglycerides'] = colesterol_realista(data['MMSE'], data['Age'], 50, 400)
data['Diabetes'] = correlated_binary(0.4, data['MMSE'], 0.4, inverse=True)
data['PhysicalActivity'] = correlated_continuous(data['MMSE'], 0.0036, 9.9874, 0.6)
data['DietQuality'] = correlated_continuous(data['MMSE'], 0, 9.99, 0.6)
data['SleepQuality'] = correlated_continuous(data['MMSE'], 4, 9.99, 0.6)
data['Depression'] = correlated_binary(0.4, data['MMSE'], 0.5, inverse=True)
data['HeadInjury'] = correlated_binary(0.3, data['MMSE'], 0.4, inverse=True)
data['Forgetfulness'] = correlated_binary(0.5, data['MMSE'], 0.5, inverse=True)
data['ADL'] = correlated_continuous(data['MMSE'], 0, 9.99, 0.6)

# Grupo 3
data['Smoking'] = correlated_binary(0.4, data['MMSE'], 0.2, inverse=True)
data['AlcoholConsumption'] = correlated_continuous(data['MMSE'], 0.0020, 19.9893, 0.4, inverse=True)
data['BMI'] = correlated_continuous(data['MMSE'], 15, 39.99, 0.5, inverse=True, dist='normal')
data['CholesterolHDL'] = correlated_continuous(data['MMSE'], 30, 100, 0.4, dist='normal')
data['Disorientation'] = correlated_binary(0.4, data['MMSE'], 0.2, inverse=True)
data['Confusion'] = correlated_binary(0.4, data['MMSE'], 0.2, inverse=True)
data['BehavioralProblems'] = correlated_binary(0.4, data['MMSE'], 0.2, inverse=True)
data['DifficultyCompletingTasks'] = correlated_binary(0.4, data['MMSE'], 0.2, inverse=True)

# Outras variáveis
data['Gender'] = np.random.binomial(1, 0.5, total_samples)
data['Ethnicity'] = np.random.randint(0, 4, total_samples)
data['PersonalityChanges'] = np.random.binomial(1, 0.2, total_samples)


def categorize_dementia(mmse_value):
    if mmse_value >= 27:
        return 'SemDemencia'
    elif mmse_value >= 24:
        return 'DemenciaMuitoLeve'
    elif mmse_value >= 19:
        return 'DemenciaLeve'
    else:
        return 'DemenciaModerada'

# Create a Pandas Series from the numpy array
mmse_series = pd.Series(data['MMSE'])
data['Diagnosis'] = mmse_series.apply(categorize_dementia)

diagnosis_mapping = {'SemDemencia': 0, 'DemenciaMuitoLeve': 1, 'DemenciaLeve': 2, 'DemenciaModerada': 3}
data['Diagnosis_encoded'] = data['Diagnosis'].map(diagnosis_mapping)

# Ruído leve nos colesteróis
data['CholesterolTotal'] += np.random.normal(0, 10, total_samples)
data['CholesterolLDL'] += np.random.normal(0, 8, total_samples)
data['CholesterolTriglycerides'] += np.random.normal(0, 15, total_samples)

# Criar DataFrame e salvar
df = pd.DataFrame(data)
df.to_csv('alzheimers_disease_data.csv', index=False)
print("Dataset realista gerado e salvo como 'alzheimers_disease_data.csv'")