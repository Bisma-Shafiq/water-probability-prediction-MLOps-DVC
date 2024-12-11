import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dagshub
import mlflow
import mlflow.metrics
import mlflow.sklearn
from sklearn.metrics import confusion_matrix

# DagsHub and MLFow experminet Tracking
dagshub.init(repo_owner='Bisma-Shafiq', repo_name='water-probability-prediction-MLOps-DVC', mlflow=True)
mlflow.set_experiment("Experiment01") # Name of the experiment

mlflow.set_tracking_uri("https://dagshub.com/Bisma-Shafiq/water-probability-prediction-MLOps-DVC.mlflow")

dataset = pd.read_csv("dataset\water_potability.csv")

#split dataset

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(dataset,test_size=0.2,random_state=42)

# Fill missing values in both train, test dataset with each column medin value
def fill_missing_value(df):
    for column in df.columns:
        if df[column].isnull().any():
            medain_value = df[column].median()
            df[column].fillna(medain_value,inplace=True)

            return df
        
# Fill missing value both in train,test dataset using median value
train_data_processed = fill_missing_value(train_data)
test_data_processed = fill_missing_value(test_data)

# Random Forest Classifier Algorithm

from sklearn.ensemble import RandomForestClassifier
import pickle

# Train, test split 
x_train = train_data_processed.drop(columns=["Potability"],axis=1)
y_train = train_data_processed["Potability"]

n_estimators=100

with mlflow.start_run():
        
       # classifier Algo
        rm = RandomForestClassifier(n_estimators=n_estimators)
        rm.fit(x_train,y_train)

        # Save file in pickle
        pickle.dump(rm,open("RMClassifierModel.pkl","wb"))

        # Prediction 
        x_test = test_data_processed.iloc[:,0:-1].values
        y_test = test_data_processed.iloc[:,-1].values

        # Import necessary metrics
        from sklearn.metrics import r2_score,accuracy_score,recall_score,f1_score,precision_score

        # Load pickle file

        model = pickle.load(open("RMClassifierModel.pkl","rb"))

        # Predict the target for test data
        y_pred = model.predict(x_test)

        # Calculate performance

        acc = accuracy_score(y_test,y_pred)
        prec = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)

        # MLFLOW Tracking

        mlflow.log_metric("accuracy",acc)
        mlflow.log_metric("recall_score",recall)
        mlflow.log_metric("precision",prec)
        mlflow.log_metric("f1_score",f1)

        # Log the number of estimators used as parameters
        mlflow.log_param("n_estimators",n_estimators)

        #Generate confusion_matrix
        cm = confusion_matrix(y_test,y_pred)
        plt.figure(figsize = (10,7))
        sns.heatmap(cm,annot = True)
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        # Save the confusion matrix plot as PNG
        plt.savefig("Confusion_matrix.png")
         
         # log the confusion matix image to mlflow
        mlflow.log_artifact("Confusion_matrix.png")

        #log the trained model
        mlflow.sklearn.log_model(rm,"RandomForestClassifier")

        # log the source code file
        mlflow.log_artifact(__file__)

        mlflow.set_tag("author","Bisma")
        mlflow.set_tag("model","GB")

        print("accuracy",acc)
        print("recall_score",recall)
        print("precision",prec)
        print("f1_score",f1)

        



