# predict.py

from ufc_model import UFCFightPredictor
import logging
import pandas as pd
import joblib

def main():
    # Configure logging for prediction
    logging.basicConfig(
        level=logging.INFO,  # You can set this to DEBUG for more details
        format='%(asctime)s %(levelname)s:%(message)s',
        handlers=[
            logging.FileHandler("predict.log"),
            logging.StreamHandler()
        ]
    )
    
    predictor = UFCFightPredictor()
    
    # Load average attributes data
    try:
        predictor.avg_attributes_df = pd.read_csv(predictor.cleaned_data_path)
        logging.info(f"Loaded average attributes from {predictor.cleaned_data_path}. Shape: {predictor.avg_attributes_df.shape}")
    except Exception as e:
        logging.error(f"Error loading average attributes data: {e}")
        raise
    
    # Load the scaler
    scaler_path = './models/scaler.joblib'
    try:
        scaler = joblib.load(scaler_path)
        logging.info(f"Scaler loaded from {scaler_path}")
    except Exception as e:
        logging.error(f"Error loading scaler: {e}")
        raise
    
    # Prediction: Predict win probability for a fighter
    fighter_name = "Conor McGregor"
    model_type = 'rf'
    
    try:
        # Retrieve fighter's averaged attributes
        fighter_row = predictor.get_fighter_row(fighter_name)
        
        # Define feature columns used during training
        feature_cols = [
            'Knockdown', 'Significant_Strike_Percent', 'Takedown_Percent',
            'Submission_Attempt', 'Ground_Control', 'win_by', 'No_of_rounds',
            'Weight Division', 'Significant_Strikes_Landed', 'Significant_Strikes_Attempted',
            'Total_Strikes_Landed', 'Total_Strikes_Attempted', 'Takedowns_Landed',
            'Takedowns_Attempted', 'Ground_Strikes_Landed', 'time_fought', 'Gender'
        ]
        
        # Extract features
        X = fighter_row[feature_cols].values.reshape(1, -1)
        
        # Apply scaling
        X_scaled = scaler.transform(X)
        logging.info(f"Applied scaling to fighter's data.")
        
        # Load the specified model
        model_path = predictor.model_paths[model_type]
        model = joblib.load(model_path)
        logging.info(f"Loaded model '{model_type}' from {model_path}")
        
        # Make prediction
        prob = model.predict_proba(X_scaled)[0][1]  # Probability of winning
        logging.info(f"Win probability for '{fighter_name}' using model '{model_type}': {prob:.4f}")
        
        print(f"Win probability for {fighter_name}: {prob * 100:.2f}%")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except FileNotFoundError as fnf:
        print(f"FileNotFoundError: {fnf}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.exception("Detailed error information:")

if __name__ == "__main__":
    main()