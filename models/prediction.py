import os
import sys
import inspect
import pandas as pd

# Adjust the import paths
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# Load models and transformers
from utils.load_models import load_models
scaler, pca, xgb_model, hypertension_model, chronic_kidney_model = load_models()

# Import the MedicalReportData model
from models.input_models import MedicalReportData

# Define feature lists
features = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
hypertension_features = ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
chronic_kidney_features = ['Age', 'Gender', 'Ethnicity', 'SocioeconomicStatus', 'BMI', 'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'FamilyHistoryKidneyDisease', 'PreviousAcuteKidneyInjury', 'SystolicBP', 'FastingBloodSugar', 'CholesterolTotal', 'ACEInhibitors', 'HeavyMetalsExposure']

# Prediction functions
def predict_diabetes(data: MedicalReportData):
    input_df = pd.DataFrame([data.dict()], columns=features)
    standardized_data = scaler.transform(input_df)
    pca_data = pca.transform(standardized_data)
    prediction = xgb_model.predict(pca_data)
    result = "Diabetes predicted" if prediction[0] == 1 else "No diabetes predicted"
    return {"prediction": result}

def predict_hypertension(data: MedicalReportData):
    input_df = pd.DataFrame([data.dict()], columns=hypertension_features)
    standardized_data = scaler.transform(input_df)
    pca_data = pca.transform(standardized_data)
    prediction = hypertension_model.predict(pca_data)
    result = "Hypertension predicted" if prediction[0] == 1 else "No hypertension predicted"
    return {"prediction": result}

def predict_chronic_kidney(data: MedicalReportData):
    input_df = pd.DataFrame([data.dict()], columns=chronic_kidney_features)
    standardized_data = scaler.transform(input_df)
    pca_data = pca.transform(standardized_data)
    prediction = chronic_kidney_model.predict(pca_data)
    result = "Chronic Kidney Disease predicted" if prediction[0] == 1 else "No Chronic Kidney Disease predicted"
    return {"prediction": result}

# Parsing function
def parse_medical_report(report_text: str) -> MedicalReportData:
    gender = 1 if "Gender: Male" in report_text else 0
    age = int(report_text.split("Age: ")[1].split()[0])
    hypertension = 0 if "Hypertension: Not present" in report_text else 1
    heart_disease = 0 if "Heart Disease: Not present" in report_text else 1
    smoking_history = 1 if "Smoking History: Smoker" in report_text else 0
    bmi = float(report_text.split("BMI): ")[1].split()[0])
    HbA1c_level = float(report_text.split("HbA1c Level: ")[1].split('%')[0])
    blood_glucose_level = int(report_text.split("Blood Glucose Level: ")[1].split()[0])

    return MedicalReportData(
        gender=gender,
        age=age,
        hypertension=hypertension,
        heart_disease=heart_disease,
        smoking_history=smoking_history,
        bmi=bmi,
        HbA1c_level=HbA1c_level,
        blood_glucose_level=blood_glucose_level
    )