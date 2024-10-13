# train_models.py

from ufc_model import UFCFightPredictor

def main():
    predictor = UFCFightPredictor()
    raw_df = predictor.load_data()
    preprocessed_df = predictor.preprocess_data(raw_df)
    predictor.train_models(preprocessed_df)
    
    # Optionally, you can calculate average attributes here if needed
    # avg_attributes = predictor.calculate_avg_attributes(years=3)
    # avg_attributes.to_csv('./data/avg_attributes.csv', index=False)

if __name__ == "__main__":
    main()