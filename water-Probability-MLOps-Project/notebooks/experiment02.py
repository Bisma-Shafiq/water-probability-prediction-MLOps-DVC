import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dagshub
import mlflow
import mlflow.metrics
import mlflow.sklearn
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# DagsHub and MLflow experiment tracking
dagshub.init(repo_owner='Bisma-Shafiq', repo_name='water-probability-prediction-MLOps-DVC', mlflow=True)
mlflow.set_experiment("Experiment02")  # Name of the experiment
mlflow.set_tracking_uri("https://dagshub.com/Bisma-Shafiq/water-probability-prediction-MLOps-DVC.mlflow")

# Load dataset
dataset = pd.read_csv("dataset/water_potability.csv")

# Split dataset
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Fill missing values in both train, test dataset with each column's median value
def fill_missing_value(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df

# Fill missing values for both train and test datasets
train_data_processed = fill_missing_value(train_data)
test_data_processed = fill_missing_value(test_data)

# Train, test split
x_train = train_data_processed.drop(columns=["Potability"], axis=1)
y_train = train_data_processed["Potability"]
x_test = test_data_processed.drop(columns=["Potability"], axis=1)
y_test = test_data_processed["Potability"]

# Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Classifier": SVC(),
    "K Nearest Neighbors": KNeighborsClassifier(),
    "decision Tree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier()
}

# Start a parent MLflow run to track overall experiment
with mlflow.start_run(run_name="Water Potability Model Experiment"):
    # Iterate over each model
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):
            # Train the model on train data
            model.fit(x_train, y_train)

            # Save model to pickle file
            model_filename = f"{model_name.replace(' ', '_')}.pkl"
            pickle.dump(model, open(model_filename, "wb"))

            # Predict the target for test data
            y_pred = model.predict(x_test)

            # Calculate performance metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log calculated metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("recall_score", recall)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("f1_score", f1)

            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Prediction")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model_name}")
            confusion_matrix_filename = f"Confusion_matrix_{model_name.replace(' ', '_')}.png"
            plt.savefig(confusion_matrix_filename)

            # Log the confusion matrix image to MLflow
            mlflow.log_artifact(confusion_matrix_filename)

            # Log the trained model to MLflow
            mlflow.sklearn.log_model(model, model_name.replace(' ', '_'))

            # Log source code (optional; replace 'your_script.py' with actual script path if needed)
            import os
            script_path = os.path.abspath(__file__) if '__file__' in globals() else "your_script.py"
            mlflow.log_artifact(script_path)

            # Add tags for better tracking
            mlflow.set_tag("author", "Bisma")
            mlflow.set_tag("model", "Multiple")

            print(f"Model {model_name} trained and logged successfully.")
