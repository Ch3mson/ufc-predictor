# ufc_model.py

import pandas as pd
import numpy as np
import joblib
import os
import re
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("ufc_model.log"),
        logging.StreamHandler()
    ]
)

class UFCFightPredictor:
    def __init__(self, data_path=None, cleaned_data_path=None, scaler_path=None, model_paths=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = data_path or os.path.join(base_dir, 'data', 'raw_total_fight_data.csv')
        self.cleaned_data_path = cleaned_data_path or os.path.join(base_dir, 'data', 'avg_attributes.csv')
        self.scaler_path = scaler_path or os.path.join(base_dir, 'models', 'scaler.joblib')

        if model_paths is None:
            self.model_paths = {
                'lr': os.path.join(base_dir, 'models', 'lr_model.joblib'),
                'rf': os.path.join(base_dir, 'models', 'rf_model.joblib'),
                'dt': os.path.join(base_dir, 'models', 'dt_model.joblib'),
                'nb': os.path.join(base_dir, 'models', 'nb_model.joblib'),
                'knn': os.path.join(base_dir, 'models', 'knn_model.joblib'),
                'svm': os.path.join(base_dir, 'models', 'svm_model.joblib')
            }
        else:
            self.model_paths = model_paths

        # Log the available model types
        logging.info(f"Available model types: {list(self.model_paths.keys())}")

        self.scaler = None
        self.models = {}
        self.avg_attributes_df = None
        logging.info("UFCFightPredictor initialized.")

    def load_data(self):
        """
        Loads raw fight data from CSV.
        """
        try:
            df = pd.read_csv(self.data_path, sep=';')
            logging.info(f"Raw data loaded from {self.data_path}. Shape: {df.shape}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def preprocess_data(self, df):
        """
        Cleans and preprocesses the fight data.
        """
        ufc = df.copy()
        logging.info("Starting data preprocessing.")

        # Drop unnecessary columns
        to_drop = ['R_REV','B_REV','R_HEAD', 'B_HEAD', 'R_BODY', 'B_BODY', 'R_LEG', 'B_LEG', 
                   'R_DISTANCE', 'B_DISTANCE', 'R_CLINCH', 'B_CLINCH', 'Referee', 'location']
        ufc = ufc.drop(to_drop, axis=1)
        logging.info(f"Dropped columns: {to_drop}")

        # Rename columns
        rename_dict = {
            'Format': 'No_of_rounds',
            'R_KD': 'R_Knockdown',
            'B_KD': 'B_Knockdown',
            'R_SIG_STR.': 'R_Significant_Strikes',
            'B_SIG_STR.': 'B_Significant_Strikes',
            'R_SIG_STR_pct': 'R_Significant_Strike_Percent',
            'B_SIG_STR_pct': 'B_Significant_Strike_Percent',
            'R_TOTAL_STR.': 'R_Total_Strikes',
            'B_TOTAL_STR.': 'B_Total_Strikes',
            'R_TD': 'R_Takedowns',
            'B_TD': 'B_Takedowns',
            'R_TD_pct': 'R_Takedown_Percent',
            'B_TD_pct': 'B_Takedown_Percent',
            'R_SUB_ATT': 'R_Submission_Attempt',
            'B_SUB_ATT': 'B_Submission_Attempt',
            'R_CTRL': 'R_Ground_Control',
            'B_CTRL': 'B_Ground_Control',
            'R_GROUND': 'R_Ground_Strikes',
            'B_GROUND': 'B_Ground_Strikes'
        }
        ufc = ufc.rename(columns=rename_dict)
        logging.info(f"Renamed columns: {rename_dict}")

        # Split 'Significant_Strikes' columns
        ufc[['R_Significant_Strikes_Landed', 'R_Significant_Strikes_Attempted']] = ufc['R_Significant_Strikes'].str.split(' of ', expand=True)
        ufc[['B_Significant_Strikes_Landed', 'B_Significant_Strikes_Attempted']] = ufc['B_Significant_Strikes'].str.split(' of ', expand=True)
        ufc = ufc.drop(['R_Significant_Strikes','B_Significant_Strikes'], axis=1)
        logging.info("Split 'Significant_Strikes' columns into landed and attempted.")

        # Split 'Total_Strikes' columns
        ufc[['R_Total_Strikes_Landed', 'R_Total_Strikes_Attempted']] = ufc['R_Total_Strikes'].str.split(' of ', expand=True)
        ufc[['B_Total_Strikes_Landed', 'B_Total_Strikes_Attempted']] = ufc['B_Total_Strikes'].str.split(' of ', expand=True)
        ufc = ufc.drop(['R_Total_Strikes','B_Total_Strikes'], axis=1)
        logging.info("Split 'Total_Strikes' columns into landed and attempted.")

        # Split 'Takedowns' columns
        ufc[['R_Takedowns_Landed', 'R_Takedowns_Attempted']] = ufc['R_Takedowns'].str.split(' of ', expand=True)
        ufc[['B_Takedowns_Landed', 'B_Takedowns_Attempted']] = ufc['B_Takedowns'].str.split(' of ', expand=True)
        ufc = ufc.drop(['R_Takedowns','B_Takedowns'], axis=1)
        logging.info("Split 'Takedowns' columns into landed and attempted.")

        # Split 'Ground_Strikes' columns
        ufc[['R_Ground_Strikes_Landed', 'R_Ground_Strikes_Attempted']] = ufc['R_Ground_Strikes'].str.split(' of ', expand=True)
        ufc[['B_Ground_Strikes_Landed', 'B_Ground_Strikes_Attempted']] = ufc['B_Ground_Strikes'].str.split(' of ', expand=True)
        ufc = ufc.drop(['R_Ground_Strikes','B_Ground_Strikes'], axis=1)
        logging.info("Split 'Ground_Strikes' columns into landed and attempted.")

        # Clean percentage columns
        percentage_cols = ['R_Significant_Strike_Percent', 'B_Significant_Strike_Percent',
                           'R_Takedown_Percent', 'B_Takedown_Percent']
        for col in percentage_cols:
            ufc[col] = ufc[col].str.replace('%', '').replace('---', '0').astype(float)
            logging.info(f"Cleaned percentage column: {col}")

        # Convert 'date' to datetime and format
        try:
            ufc['date'] = pd.to_datetime(ufc['date'], format="%B %d, %Y").dt.strftime("%Y-%m-%d")
            logging.info("Converted 'date' to datetime and formatted.")
        except Exception as e:
            logging.error(f"Error converting 'date': {e}")
            raise

        # Filter by date
        limit_date = '2001-04-01'
        original_shape = ufc.shape
        ufc = ufc[ufc['date'] > limit_date]
        logging.info(f"Filtered data by date > {limit_date}. Rows before: {original_shape[0]}, after: {ufc.shape[0]}.")

        # Drop rows with NaN in 'Winner'
        ufc = ufc.dropna(subset=['Winner'])
        logging.info(f"Dropped rows with NaN in 'Winner'. New shape: {ufc.shape}")

        # Calculate 'time_fought'
        def calculate_total_duration(value):
            try:
                parts = value.split(':')
                minutes = int(parts[0])
                seconds = int(parts[1])
                total_seconds = minutes * 60 + seconds
                return total_seconds
            except Exception:
                return 0  # Or any default value

        ufc['last_round_time'] = ufc['last_round_time'].apply(calculate_total_duration)
        ufc['time_fought'] = (ufc['last_round'] - 1) * 5 * 60 + ufc['last_round_time']
        ufc = ufc.drop(['last_round','last_round_time'], axis=1)
        logging.info("Calculated 'time_fought' and dropped 'last_round' and 'last_round_time'.")

        # Drop fights won by DQ
        ufc = ufc[ufc['win_by'] != 'DQ']
        logging.info("Dropped fights won by DQ.")

        # Process 'R_fighter' and 'B_fighter' data
        R_subset = ['R_fighter','R_Knockdown','R_Significant_Strike_Percent','R_Takedown_Percent', 
                    'R_Submission_Attempt','R_Ground_Control', 'win_by', 'No_of_rounds', 
                    'date', 'Fight_type', 'R_Significant_Strikes_Landed', 
                    'R_Significant_Strikes_Attempted', 'R_Total_Strikes_Landed', 
                    'R_Total_Strikes_Attempted', 'R_Takedowns_Landed', 'R_Takedowns_Attempted', 
                    'R_Ground_Strikes_Landed', 'time_fought','Winner']
        R_df = ufc[R_subset].rename(columns=lambda x: x.replace('R_', ''))
        R_df['Winner'] = np.where(R_df['Winner'] == R_df['fighter'], 1, 0)

        B_subset = ['B_fighter', 'B_Knockdown', 'B_Significant_Strike_Percent', 'B_Takedown_Percent', 
                    'B_Submission_Attempt','B_Ground_Control', 'win_by', 'No_of_rounds', 
                    'date', 'Fight_type', 'B_Significant_Strikes_Landed',
                    'B_Significant_Strikes_Attempted', 'B_Total_Strikes_Landed', 
                    'B_Total_Strikes_Attempted', 'B_Takedowns_Landed', 'B_Takedowns_Attempted', 
                    'B_Ground_Strikes_Landed', 'time_fought', 'Winner']
        B_df = ufc[B_subset].rename(columns=lambda x: x.replace('B_', ''))
        B_df['Winner'] = np.where(B_df['Winner'] == B_df['fighter'], 1, 0)

        # Combine R_df and B_df
        new_df = pd.concat([R_df, B_df])
        ufc = new_df.sort_values(by='date', ascending=False).reset_index(drop=True)
        logging.info(f"Combined R_df and B_df. New shape: {ufc.shape}")

        # Convert 'Ground_Control' to total duration
        ufc['Ground_Control'] = ufc['Ground_Control'].apply(calculate_total_duration)
        logging.info("Converted 'Ground_Control' to total duration.")

        # Map 'win_by' to numerical values
        mapping = {
            'KO/TKO': 10,
            'Submission': 9,
            "TKO - Doctor's Stoppage": 9,
            'Decision - Unanimous': 9,
            'Decision - Majority': 8,
            'Decision - Split': 7
        }
        try:
            ufc['win_by'] = ufc['win_by'].replace(mapping).astype(float)
            logging.info("Mapped 'win_by' to numerical values.")
        except Exception as e:
            logging.error(f"Error mapping 'win_by': {e}")
            raise

        # Clean 'Fight_type'
        fight_type_cleaning = {
            'Bout': '',
            'Title': '',
            'Tournament': '',
            'Ultimate Fighter': '',
            'UFC': '',
            'Interim': '',
            'Brazil': '',
            'America': '',
            'China': '',
            'TUF': '',
            'Australia': '',
            'Nations': '',
            'Canada': '',
            'vs.': '',
            'UK': '',
            'Latin': '',
            "Women's": 'W'
        }
        ufc['Fight_type'] = ufc['Fight_type'].replace(fight_type_cleaning, regex=True)
        ufc['Fight_type'] = ufc['Fight_type'].apply(lambda x: re.sub(r'\d+', ' ', x)).str.strip()
        logging.info("Cleaned 'Fight_type' column.")

        # Map 'Fight_type' to numerical weight divisions using .map()
        weight_mapping = {
            'W Strawweight': 115,
            'W Flyweight': 125,
            'W Bantamweight': 135,
            'W Featherweight': 145,
            'Flyweight': 125,
            'Bantamweight': 135,
            'Featherweight': 145,
            'Lightweight': 155,
            'Welterweight': 170,
            'Middleweight': 185,
            'Light Heavyweight': 205,
            'Heavyweight': 265,
        }
        ufc['Fight_type'] = ufc['Fight_type'].map(weight_mapping)
        logging.info("Mapped 'Fight_type' to numerical weight divisions.")

        # Check for unmapped 'Fight_type' values and handle them
        unmapped_fight_types = ufc[ufc['Fight_type'].isna()]['Fight_type'].unique()
        if len(unmapped_fight_types) > 0:
            logging.warning(f"Unmapped 'Fight_type' values found: {unmapped_fight_types}")
            # Handle unmapped entries by dropping them
            ufc = ufc.dropna(subset=['Fight_type'])
            logging.info(f"Dropped rows with unmapped 'Fight_type'. New shape: {ufc.shape}")

        # Rename 'Fight_type' to 'Weight Division' after mapping
        ufc = ufc.rename(columns={'Fight_type': 'Weight Division'})
        logging.info("Renamed 'Fight_type' to 'Weight Division'.")

        # Assign Gender based on numerical weight divisions
        def assign_gender(row):
            women_weights = [115, 125, 135, 145]  # Numerical representations for women's weight classes
            if row['Weight Division'] in women_weights:
                return 0
            else:
                return 1

        ufc['Gender'] = ufc.apply(assign_gender, axis=1)
        logging.info("Assigned 'Gender' based on 'Weight Division'.")

        # Ensure 'Weight Division' is integer and handle any remaining NaNs
        ufc['Weight Division'] = ufc['Weight Division'].astype(int)
        logging.info("Ensured 'Weight Division' is integer.")

        # Process 'No_of_rounds' to ensure it's numeric
        # Replace any non-digit characters and extract the first number
        ufc['No_of_rounds'] = ufc['No_of_rounds'].astype(str).str.extract(r'(\d+)')
        ufc['No_of_rounds'] = pd.to_numeric(ufc['No_of_rounds'], errors='coerce').fillna(3).astype(int)  # Replace 3 with appropriate default
        logging.info(f"Processed 'No_of_rounds'. Unique values after extraction: {ufc['No_of_rounds'].unique()}")

        # Move 'date' to first column
        date_column = ufc['date']
        ufc = ufc.drop(columns=['date'])
        ufc.insert(0, 'date', date_column)
        logging.info("Moved 'date' to first column.")

        # Convert split strike columns to numeric
        strike_columns = [
            'Significant_Strikes_Landed', 'Significant_Strikes_Attempted',
            'Total_Strikes_Landed', 'Total_Strikes_Attempted',
            'Takedowns_Landed', 'Takedowns_Attempted',
            'Ground_Strikes_Landed'
        ]

        for col in strike_columns:
            ufc[col] = pd.to_numeric(ufc[col], errors='coerce').fillna(0).astype(int)
            logging.info(f"Converted '{col}' to numeric.")

        # Check for any remaining non-numeric columns in columns_to_standardize
        # Exclude 'date', 'Gender', and 'Winner' from standardization
        columns_to_standardize = ufc.columns[2:-1].tolist()  # Exclude 'date', 'Winner'
        for col in columns_to_standardize:
            if not pd.api.types.is_numeric_dtype(ufc[col]):
                logging.warning(f"Non-numeric column found in standardization list: {col}")
                # Attempt to convert to numeric, coerce errors to NaN
                ufc[col] = pd.to_numeric(ufc[col], errors='coerce')
                # Fill NaN with column mean or a default value
                ufc[col] = ufc[col].fillna(ufc[col].mean())
                logging.info(f"Converted '{col}' to numeric and filled NaNs with mean.")

        # Copy to ufc2
        ufc2 = ufc.copy()
        logging.info("Copied preprocessed data to 'ufc2'.")

        # Drop 'date' and 'fighter' from ufc2 as they are not features
        ufc2 = ufc2.drop(['date', 'fighter'], axis=1)
        logging.info("Dropped 'date' and 'fighter' from preprocessed data.")

        # Standardize features (excluding 'Winner')
        feature_cols = ufc2.columns.tolist()
        feature_cols.remove('Winner')  # Exclude target variable from scaling
        scaler = StandardScaler()
        scaler.fit(ufc2[feature_cols])
        ufc2[feature_cols] = scaler.transform(ufc2[feature_cols])
        logging.info("Standardized numerical features.")

        # Save cleaned data and scaler
        try:
            ufc2.to_csv(self.cleaned_data_path, index=False)
            os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
            joblib.dump(scaler, self.scaler_path)
            logging.info(f"Saved cleaned data to {self.cleaned_data_path} and scaler to {self.scaler_path}.")
        except Exception as e:
            logging.error(f"Error saving cleaned data or scaler: {e}")
            raise

        self.avg_attributes_df = ufc2
        logging.info("Preprocessing completed successfully.")

        # Debugging: Check if 'Weight Division' exists and contains numerical values
        logging.info(f"Columns in DataFrame after preprocessing: {ufc.columns.tolist()}")
        logging.info(f"Sample data:\n{ufc[['Weight Division', 'Gender', 'No_of_rounds']].head()}")

        return ufc2

    def train_models(self, df):
        """
        Trains multiple machine learning models and saves them.

        Parameters:
        - df (pd.DataFrame): Preprocessed DataFrame.
        """
        logging.info("Starting model training.")

        # Define feature columns and target
        feature_cols = [
            'Knockdown', 'Significant_Strike_Percent', 'Takedown_Percent',
            'Submission_Attempt', 'Ground_Control', 'win_by', 'No_of_rounds',
            'Weight Division', 'Significant_Strikes_Landed', 'Significant_Strikes_Attempted',
            'Total_Strikes_Landed', 'Total_Strikes_Attempted', 'Takedowns_Landed',
            'Takedowns_Attempted', 'Ground_Strikes_Landed', 'time_fought', 'Gender'
        ]
        target_col = 'Winner'

        # Ensure all feature columns are present
        missing_features = set(feature_cols) - set(df.columns)
        if missing_features:
            logging.error(f"Missing feature columns: {missing_features}")
            raise ValueError(f"Missing feature columns: {missing_features}")

        X = df[feature_cols]
        y = df[target_col]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logging.info(f"Split data into train and test sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Initialize models
        models = {
            'lr': LogisticRegression(max_iter=1000),
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'dt': DecisionTreeClassifier(random_state=42),
            'nb': GaussianNB(),
            'knn': KNeighborsClassifier(),
            'svm': SVC(probability=True, random_state=42)
        }

        # Train and evaluate models
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                logging.info(f"Model: {name.upper()}")
                logging.info(f"Accuracy: {acc:.4f}")
                logging.info(f"Confusion Matrix:\n{cm}\n")

                # Save the model
                model_path = self.model_paths.get(name)
                if model_path:
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    joblib.dump(model, model_path)
                    logging.info(f"Model '{name}' saved to {model_path}\n")
                else:
                    logging.warning(f"No path specified for model '{name}'. Skipping saving.\n")
            except Exception as e:
                logging.error(f"Error training model '{name}': {e}")

        logging.info("Model training completed successfully.")

    def calculate_avg_attributes(self, years=3):
        """
        Calculates average attributes over the past specified years.

        Parameters:
        - years (int): Number of years to consider for averaging.

        Returns:
        - pd.DataFrame: DataFrame with average attributes.
        """
        logging.info(f"Calculating average attributes over the past {years} years.")
        try:
            current_year = pd.to_datetime('today').year
            cutoff_date = f"{current_year - years}-01-01"
            # Assuming 'time_fought' represents a date in YYYY-MM-DD format
            recent_fights = self.avg_attributes_df[self.avg_attributes_df['time_fought'] >= cutoff_date]
            logging.info(f"Filtered fights after cutoff date {cutoff_date}. Shape: {recent_fights.shape}")

            # Select only numeric columns excluding 'Winner'
            numeric_cols = recent_fights.select_dtypes(include=[np.number]).columns.tolist()
            if 'Winner' in numeric_cols:
                numeric_cols.remove('Winner')
            logging.info(f"Numeric columns to average: {numeric_cols}")

            # Group by 'fighter' and calculate mean
            avg_attributes = recent_fights.groupby('fighter')[numeric_cols].mean().reset_index()
            logging.info("Calculated average attributes successfully.")
            return avg_attributes
        except Exception as e:
            logging.error(f"Error calculating average attributes: {e}")
            raise

    def get_fighter_row(self, fighter_name):
        """
        Retrieves the row corresponding to the specified fighter.

        Parameters:
        - fighter_name (str): Name of the fighter.

        Returns:
        - pd.Series: Row of the fighter's attributes.

        Raises:
        - ValueError: If the fighter is not found.
        """
        # Since 'avg_attributes_df' no longer includes 'date' and 'Winner', ensure we have the correct DataFrame
        fighter_row = self.avg_attributes_df[self.avg_attributes_df['fighter'] == fighter_name]
        if fighter_row.empty:
            logging.error(f"Fighter '{fighter_name}' not found in the dataset.")
            raise ValueError(f"Fighter '{fighter_name}' not found in the dataset.")
        logging.info(f"Retrieved data for fighter '{fighter_name}'.")
        return fighter_row.iloc[0]

    def get_win_probability(self, fighter_name, model_type='rf'):
        """
        Calculates the win probability for a given fighter using the specified model.

        Parameters:
        - fighter_name (str): Name of the fighter.
        - model_type (str): Type of model to use ('lr', 'rf', etc.).

        Returns:
        - float: Win probability between 0 and 1.

        Raises:
        - ValueError: If the fighter is not found in the dataset.
        - AttributeError: If the selected model does not support probability predictions.
        """
        try:
            # Debugging: Log the available model types
            logging.debug(f"Available model types: {list(self.model_paths.keys())}")

            fighter_row = self.get_fighter_row(fighter_name)
            
            # Define the feature columns used during training
            feature_cols = [
                'Knockdown', 'Significant_Strike_Percent', 'Takedown_Percent',
                'Submission_Attempt', 'Ground_Control', 'win_by', 'No_of_rounds',
                'Weight Division', 'Significant_Strikes_Landed', 'Significant_Strikes_Attempted',
                'Total_Strikes_Landed', 'Total_Strikes_Attempted', 'Takedowns_Landed',
                'Takedowns_Attempted', 'Ground_Strikes_Landed', 'time_fought', 'Gender'
            ]
            
            # Ensure that only the feature columns are selected
            X = fighter_row[feature_cols].values.reshape(1, -1)
            
            # Load the model
            model = joblib.load(self.model_paths[model_type])
            prob = model.predict_proba(X)[0][1]  # Probability of winning
            logging.info(f"Win probability for '{fighter_name}' using model '{model_type}': {prob:.4f}")
            return prob
        except KeyError:
            logging.error(f"Model type '{model_type}' is not recognized.")
            raise ValueError(f"Model type '{model_type}' is not recognized.")
        except AttributeError as e:
            logging.error(f"Model '{model_type}' does not support probability predictions: {e}")
            raise AttributeError(f"Model '{model_type}' does not support probability predictions.")
        except Exception as e:
            logging.error(f"Error calculating win probability: {e}")
            raise

    def match_probability(self, fighter_A, fighter_B, model_type='rf'):
        """
        Calculates the match probability between two fighters using the specified model.

        Parameters:
        - fighter_A (str): Name of the first fighter.
        - fighter_B (str): Name of the second fighter.
        - model_type (str): Type of model to use ('lr', 'rf', etc.).

        Returns:
        - tuple: (prob_A, prob_B) probabilities for fighter A and fighter B.

        Raises:
        - ValueError: If either fighter is not found in the dataset.
        - AttributeError: If the selected model does not support probability predictions.
        """
        try:
            fighter_A_row = self.get_fighter_row(fighter_A)
            fighter_B_row = self.get_fighter_row(fighter_B)

            # Define the feature columns used during training
            feature_cols = [
                'Knockdown', 'Significant_Strike_Percent', 'Takedown_Percent',
                'Submission_Attempt', 'Ground_Control', 'win_by', 'No_of_rounds',
                'Weight Division', 'Significant_Strikes_Landed', 'Significant_Strikes_Attempted',
                'Total_Strikes_Landed', 'Total_Strikes_Attempted', 'Takedowns_Landed',
                'Takedowns_Attempted', 'Ground_Strikes_Landed', 'time_fought', 'Gender'
            ]
            
            # Combine features appropriately; here we average them
            combined_features = (fighter_A_row[feature_cols] + fighter_B_row[feature_cols]) / 2
            X = combined_features.values.reshape(1, -1)

            model = joblib.load(self.model_paths[model_type])
            prob = model.predict_proba(X)[0]
            prob_A, prob_B = prob[1], prob[0]  # Assuming prob[1] is fighter A's win probability
            logging.info(f"Match probability between '{fighter_A}' and '{fighter_B}' using model '{model_type}': {prob_A:.4f}, {prob_B:.4f}")
            return prob_A, prob_B
        except KeyError:
            logging.error(f"Model type '{model_type}' is not recognized.")
            raise ValueError(f"Model type '{model_type}' is not recognized.")
        except AttributeError as e:
            logging.error(f"Model '{model_type}' does not support probability predictions: {e}")
            raise AttributeError(f"Model '{model_type}' does not support probability predictions.")
        except Exception as e:
            logging.error(f"Error calculating match probability: {e}")
            raise