import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import pickle

def load_models():
    # Load the pre-trained models and transformers
    with open('static/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('static/pca.pkl', 'rb') as f:
        pca = pickle.load(f)

    with open('static/diabetes_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    with open('static/hypertension_model.pkl', 'rb') as f:
        hypertension_model = pickle.load(f)

    with open('static/chronic_kidney_model.pkl', 'rb') as f:
        chronic_kidney_model = pickle.load(f)
    
    return scaler, pca, xgb_model, hypertension_model, chronic_kidney_model
