import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from fastapi import FastAPI, File, UploadFile, HTTPException
from models.input_models import InputData, HypertensionInputData, ChronicKidneyInputData
from models.prediction import predict_diabetes, predict_hypertension, predict_chronic_kidney, parse_medical_report
import pdfplumber
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/predict-diabetes/")
async def predict_diabetes_endpoint(file: UploadFile = File(...)):
    try:
        pdf_path = f"/tmp/{file.filename}"
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        # Extract text from the PDF
        with pdfplumber.open(pdf_path) as pdf:
            report_text = ""
            for page in pdf.pages:
                report_text += page.extract_text() + "\n"

        # Parse the extracted text
        parsed_data = parse_medical_report(report_text)
    
        # Get the prediction
        prediction = predict_diabetes(parsed_data)
        
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_hypertension")
def predict_hypertension_endpoint(data: HypertensionInputData):
    return predict_hypertension(data)

@app.post("/predict_chronic_kidney")
def predict_chronic_kidney_endpoint(data: ChronicKidneyInputData):
    return predict_chronic_kidney(data)

@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    # Save the uploaded file
    pdf_path = f"/tmp/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    # Extract text from the PDF
    with pdfplumber.open(pdf_path) as pdf:
        report_text = ""
        for page in pdf.pages:
            report_text += page.extract_text() + "\n"
    
    # Parse the extracted text
    parsed_data = parse_medical_report(report_text)
    
    # Return the parsed data
    return parsed_data