import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import pandas as pd
from utils.load_models import load_models

# Load all models and transformers
scaler, pca, xgb_model, hypertension_model, chronic_kidney_model = load_models()

# Define feature lists
features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
hypertension_features = ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
chronic_kidney_features = ['Age', 'Gender', 'Ethnicity', 'SocioeconomicStatus', 'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'FamilyHistoryKidneyDisease', 'PreviousAcuteKidneyInjury', 'SystolicBP', 'FastingBloodSugar', 'CholesterolTotal', 'ACEInhibitors', 'HeavyMetalsExposure']

def predict_diabetes(data):
    input_df = pd.DataFrame([data.dict()], columns=features)
    standardized_data = scaler.transform(input_df)
    pca_data = pca.transform(standardized_data)
    prediction = xgb_model.predict(pca_data)
    result = "Diabetes predicted" if prediction[0] == 1 else "No diabetes predicted"
    return {"prediction": result}

def predict_hypertension(data):
    input_df = pd.DataFrame([data.dict()], columns=hypertension_features)
    standardized_data = scaler.transform(input_df)
    pca_data = pca.transform(standardized_data)
    prediction = hypertension_model.predict(pca_data)
    result = "Hypertension predicted" if prediction[0] == 1 else "No hypertension predicted"
    return {"prediction": result}

def predict_chronic_kidney(data):
    input_df = pd.DataFrame([data.dict()], columns=chronic_kidney_features)
    standardized_data = scaler.transform(input_df)
    pca_data = pca.transform(standardized_data)
    prediction = chronic_kidney_model.predict(pca_data)
    result = "Chronic Kidney Disease predicted" if prediction[0] == 1 else "No Chronic Kidney Disease predicted"
    return {"prediction": result}
