import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from fastapi import FastAPI
from models.input_models import InputData, HypertensionInputData, ChronicKidneyInputData
from models.prediction import predict_diabetes, predict_hypertension, predict_chronic_kidney

app = FastAPI()

# Define the endpoints
@app.post("/predict")
def predict_diabetes_endpoint(data: InputData):
    return predict_diabetes(data)

@app.post("/predict_hypertension")
def predict_hypertension_endpoint(data: HypertensionInputData):
    return predict_hypertension(data)

@app.post("/predict_chronic_kidney")
def predict_chronic_kidney_endpoint(data: ChronicKidneyInputData):
    return predict_chronic_kidney(data)
