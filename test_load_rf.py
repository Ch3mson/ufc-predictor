import joblib
import os

model_path = '/Users/bensonyan/Desktop/projects/ufc/models/rf_model.joblib'

if os.path.exists(model_path):
    try:
        rf_model = joblib.load(model_path)
        print("Random Forest model loaded successfully.")
    except Exception as e:
        print(f"Error loading Random Forest model: {e}")
else:
    print(f"Model file does not exist at {model_path}")