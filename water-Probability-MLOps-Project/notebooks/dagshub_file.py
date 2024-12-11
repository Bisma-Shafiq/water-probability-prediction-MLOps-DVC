import dagshub
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/Bisma-Shafiq/water-probability-prediction-MLOps-DVC.mlflow")

dagshub.init(repo_owner='Bisma-Shafiq', repo_name='water-probability-prediction-MLOps-DVC', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)