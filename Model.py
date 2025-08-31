import pandas as pd
import logging 
import yaml
import os 
import sys
import json
import time 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import mean_squared_error, r2_score
from Data_loader import DataLoader

# Fix: Use the correct config path
config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


def get_model_type(Model_type):
    
    models_type = {
        "Linear Regression": LinearRegression, 
        "Random Forest Regressor": RandomForestRegressor,
        "Gradient Boosting Regressor": GradientBoostingRegressor,
        "XGBoost Regressor": XGBRegressor,
        "Support Vector Regressor": SVR,
        "K-Nearest Neighbors Regressor": KNeighborsRegressor,
        "Decision Tree Regressor": DecisionTreeRegressor
    }

    return models_type.get(Model_type)

def save_results_to_json(results, model_name):
    try:
        os.makedirs('results', exist_ok=True)
        clean_model_name = model_name.replace(' ', '_').replace('-','_')
        filename = f"results/R1_{clean_model_name}.json"

        with open(filename, 'w') as file:
            json.dump(results, file, indent=2)
        logging.info(f"Results saved to {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error saving results to json: {e}")
        sys.exit(1)

def get_latest_result_file():
    try:
        result_dir = 'results'
        if not os.path.exists(result_dir):
            return None

        json_files = [f for f in os.listdir(result_dir) if f.endswith('.json')]

        if not json_files:
            return None

        json_files.sort(key=lambda x: os.path.getmtime(os.path.join(result_dir, x), reverse=True))

        latest_file = os.path.join(result_dir, json_files[0])
        logging.info(f"Latest result file: {latest_file}")
        return latest_file

    except Exception as e:
        logging.error(f"Error getting latest result file:{e}")
        return None

def preprocess_data():
    try:
        # Fix: Use the same config_path variable
        df = DataLoader(config_path).load_data()
        logging.info(f"Successfully loaded data")

        df_cleaned = df.drop(columns=['Rank','Name','Percentage_Change_Clean'])
        logging.info(f"Successfully preprocessed data")
        return df_cleaned

    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        sys.exit(1)
    

def split_data_and_train_model():
    try: 
        data_processed = preprocess_data()
        
        numerical_features = config['Features']['numerical']
        categorical_features = config['Features']['categorical']
        target_feature = config['Features']['target']

        X = data_processed[numerical_features + categorical_features]
        y = data_processed[target_feature]

        X = pd.get_dummies(X, columns=categorical_features)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        selected_models = config['Selected_Models']
        model_class = get_model_type(selected_models)

        if model_class:
            if 'random_state' in model_class.__init__.__code__.co_varnames:
                model = model_class(random_state=42)
            else: 
                model = model_class()
            logging.info(f"Model {selected_models} is loaded successfully")

            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results = {
                "Model": selected_models,
                "Data_info": {
                    "Training_samples": len(X_train),
                    "Testing_samples": len(X_test),
                    "Total_samples": len(X_train) + len(X_test)
                },
                "Performance_Metrics": {
                    "Mean_Squared_Error": float(mse),
                    "R2_Score": float(r2)
                },
                "Training_Time":{
                    "Training_Time_Seconds": round(training_time, 2)}
            }
            result_file = save_results_to_json(results, selected_models)
            logging.info(f"Results saved to {result_file}")
            
            return model, X_test, y_test, y_pred, mse, r2, results, result_file
            
    except Exception as e:
        logging.error(f"Error training model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    model, X_test, y_test, y_pred, mse, r2, results, result_file = split_data_and_train_model()
    print(f"Model trained successfully! Results saved to: {result_file}")
