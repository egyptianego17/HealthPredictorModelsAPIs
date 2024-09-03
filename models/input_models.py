from pydantic import BaseModel

# Define the input data model for diabetes prediction
class InputData(BaseModel):
    gender: int
    age: int
    hypertension: int
    heart_disease: int
    smoking_history: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int

# Define the input data model for hypertension prediction
class HypertensionInputData(BaseModel):
    male: int
    age: int
    currentSmoker: int
    cigsPerDay: float
    BPMeds: float
    diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float

# Define the input data model for chronic kidney disease prediction
class ChronicKidneyInputData(BaseModel):
    Age: int
    Gender: int
    Ethnicity: int
    SocioeconomicStatus: int
    BMI: float
    Smoking: int
    AlcoholConsumption: float
    PhysicalActivity: float
    DietQuality: float
    SleepQuality: float
    FamilyHistoryKidneyDisease: int
    PreviousAcuteKidneyInjury: int
    SystolicBP: int
    FastingBloodSugar: float
    CholesterolTotal: float
    ACEInhibitors: int
    HeavyMetalsExposure: int
