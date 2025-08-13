import pandas as pd
import logging 
import yaml
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import mean_squared_error, r2_score
import json
from Data_loader import DataLoader
import sys

with open('config.yml', 'r') as file:
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

def save_results_to_json(results):
    try:
        with open('results.json', 'w') as file:
            json.dump(results, file)
        logging.info(f"Results saved to results.json")
    except Exception as e:
        logging.error(f"Error saving results to json: {e}")
        sys.exit(1)

def preprocess_data():
    try:
        df = DataLoader('config.yml').load_data()
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

            model.fit(X_train, y_train)

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
                }
            }
            save_results_to_json(results)
            logging.info(f"Results saved to results.json")
            
            return model, X_test, y_test, y_pred, mse, r2, results  # Add return statement
            
    except Exception as e:
        logging.error(f"Error training model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    model, X_test, y_test, y_pred, mse, r2, results = split_data_and_train_model()
