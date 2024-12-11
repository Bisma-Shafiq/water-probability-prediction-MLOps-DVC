import os 
import pandas as pd
import yaml
import mlflow
import mlflow.metrics
import mlflow.sklearn
import pickle
from sklearn.ensemble import RandomForestClassifier

def load_params(params_path: str) -> int:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params["model_building"]["n_estimators"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {params_path}: {e}")

def load_data(data_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {e}")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X, y)
        return clf
    except Exception as e:
        raise Exception(f"Error training model: {e}")

def save_model(model: RandomForestClassifier, model_name: str) -> None:
    try:
        with open(model_name, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model to {model_name}: {e}")

def main():
    try:
        params_path = "params.yaml"
        data_path = "./data/processed/train_processed.csv"
        model_name = "models/model.pkl"
        # Ensure the directory for saving the model exists
        os.makedirs(os.path.dirname(model_name), exist_ok=True)

        n_estimators = load_params(params_path)
        train_data = load_data(data_path)
        X_train, y_train = prepare_data(train_data)


        with mlflow.start_run():
                # Log parameters
                mlflow.log_param("n_estimators", n_estimators)

                # Train and save model
                model = train_model(X_train, y_train, n_estimators)
                save_model(model, model_name)

                # Log the model and its metrics
                mlflow.sklearn.log_model(model, "model")
                print("Model trained and logged to MLflow successfully!")
    except Exception as e:
            
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()