from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import logging
import os

app = Flask(__name__)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'avg_attributes.csv')

# Load Scaler
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
try:
    scaler = joblib.load(SCALER_PATH)
    logging.info(f"Scaler loaded successfully from {SCALER_PATH}")
except Exception as e:
    logging.error(f"Error loading scaler: {e}")
    scaler = None

# Load Models
model_paths = {
    'lr': os.path.join(MODELS_DIR, 'lr_model.joblib'),
    'rf': os.path.join(MODELS_DIR, 'rf_model.joblib'),
    'dt': os.path.join(MODELS_DIR, 'dt_model.joblib'),
    'nb': os.path.join(MODELS_DIR, 'nb_model.joblib'),
    'knn': os.path.join(MODELS_DIR, 'knn_model.joblib'),
    'svm': os.path.join(MODELS_DIR, 'svm_model.joblib')
}

models = {}
for key, path in model_paths.items():
    try:
        models[key] = joblib.load(path)
        logging.info(f"Model '{key}' loaded successfully from {path}")
    except Exception as e:
        logging.error(f"Error loading model '{key}': {e}")
        models[key] = None

# Load Processed Data
try:
    avg_attributes_df = pd.read_csv(DATA_PATH)
    logging.info(f"Loaded average attributes from {DATA_PATH}. Shape: {avg_attributes_df.shape}")
except Exception as e:
    logging.error(f"Error loading average attributes data: {e}")
    avg_attributes_df = None

# Feature Columns (Ensure these match the training phase)
FEATURE_COLS = [
    'Knockdown', 'Significant_Strike_Percent', 'Takedown_Percent',
    'Submission_Attempt', 'Ground_Control', 'win_by', 'No_of_rounds',
    'Weight Division', 'Significant_Strikes_Landed', 'Significant_Strikes_Attempted',
    'Total_Strikes_Landed', 'Total_Strikes_Attempted', 'Takedowns_Landed',
    'Takedowns_Attempted', 'Ground_Strikes_Landed', 'time_fought', 'Gender'
]

# Helper Function to Retrieve Fighter's Averaged Attributes
def get_fighter_attributes(fighter_name):
    if avg_attributes_df is None:
        logging.error("Average attributes data not loaded.")
        return None
    
    fighter_data = avg_attributes_df[avg_attributes_df['fighter'] == fighter_name]
    if fighter_data.empty:
        logging.error(f"Fighter '{fighter_name}' not found in the dataset.")
        return None
    
    fighter_row = fighter_data.iloc[0][FEATURE_COLS]
    return fighter_row.values.reshape(1, -1)

# API Endpoint for Single Fighter Win Probability
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    fighter_name = data.get('fighter_name')
    model_type = data.get('model_type', 'rf')  # Default to 'rf' if not specified
    
    logging.info(f"Received prediction request for fighter '{fighter_name}' using model '{model_type}'")
    
    if model_type not in models or models[model_type] is None:
        logging.error(f"Model type '{model_type}' is not available.")
        return jsonify({'error': f"Model type '{model_type}' is not available."}), 400
    
    fighter_attributes = get_fighter_attributes(fighter_name)
    if fighter_attributes is None:
        return jsonify({'error': f"Fighter '{fighter_name}' not found."}), 404
    
    # Scale the attributes
    if scaler is not None:
        fighter_attributes_scaled = scaler.transform(fighter_attributes)
    else:
        logging.error("Scaler is not loaded.")
        return jsonify({'error': "Scaler is not loaded."}), 500
    
    # Predict probability
    try:
        model = models[model_type]
        probability = model.predict_proba(fighter_attributes_scaled)[0][1]  # Probability of winning
        logging.info(f"Predicted win probability for '{fighter_name}': {probability:.4f}")
        return jsonify({
            'fighter_name': fighter_name,
            'model_type': model_type,
            'win_probability': round(probability * 100, 2)
        }), 200
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': "An error occurred during prediction."}), 500

# API Endpoint for Match Probability Between Two Fighters
@app.route('/match_predict', methods=['POST'])
def match_predict():
    data = request.get_json()
    fighter_A = data.get('fighter_A')
    fighter_B = data.get('fighter_B')
    model_type = data.get('model_type', 'rf')  # Default to 'rf' if not specified
    
    logging.info(f"Received match prediction request between '{fighter_A}' and '{fighter_B}' using model '{model_type}'")
    
    if model_type not in models or models[model_type] is None:
        logging.error(f"Model type '{model_type}' is not available.")
        return jsonify({'error': f"Model type '{model_type}' is not available."}), 400
    
    # Retrieve attributes for both fighters
    attributes_A = get_fighter_attributes(fighter_A)
    attributes_B = get_fighter_attributes(fighter_B)
    
    if attributes_A is None or attributes_B is None:
        return jsonify({'error': f"One or both fighters not found in the dataset."}), 404
    
    # Scale the attributes
    if scaler is not None:
        attributes_A_scaled = scaler.transform(attributes_A)
        attributes_B_scaled = scaler.transform(attributes_B)
    else:
        logging.error("Scaler is not loaded.")
        return jsonify({'error': "Scaler is not loaded."}), 500
    
    # Predict probabilities
    try:
        model = models[model_type]
        prob_A = model.predict_proba(attributes_A_scaled)[0][1]
        prob_B = model.predict_proba(attributes_B_scaled)[0][1]
        
        # Normalize probabilities
        total_prob = prob_A + prob_B
        probability_A = round((prob_A / total_prob) * 100, 2) if total_prob > 0 else 0
        probability_B = round((prob_B / total_prob) * 100, 2) if total_prob > 0 else 0
        
        logging.info(f"Predicted match probabilities: '{fighter_A}': {probability_A}%, '{fighter_B}': {probability_B}%")
        
        return jsonify({
            'fighter_A': fighter_A,
            'fighter_B': fighter_B,
            'model_type': model_type,
            'probability_A': probability_A,
            'probability_B': probability_B
        }), 200
    except Exception as e:
        logging.error(f"Error during match prediction: {e}")
        return jsonify({'error': "An error occurred during match prediction."}), 500

# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is running.'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)