import os
import sys

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger.logging import logging
from src.exception.exception import CustomeException
from src.utils.utils import load_object

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

@dataclass 
class ModelEvaluationConfig:
    pass

class ModelEvaluation:
    
    def __init__(self):
        logging.info("Evaluation started")

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual ,pred)
        r2 = r2_score(actual , pred)
        logging.info("Evaluation metrics captured")
        return rmse, mae, r2
    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:,:-1], test_array[:,-1])

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            mlflow.set_registry_uri("") # uri of cloud location where model is to be registered
            logging.info("Model is registred")

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predictions = model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, predictions) 

                #mlflow.log_param()
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2_score", r2)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="first_model")
                else:
                    mlflow.sklearn.log_model(model, "model")


        except Exception as e:
            raise CustomeException(e,sys)
        
        
        
        
    
    
    