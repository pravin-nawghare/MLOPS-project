import os
import sys

import numpy as np
import pandas as pd

from src.logger.logging import logging
from src.exception.exception import CustomeException
from src.utils.utils import save_object, evaluate_model

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge 
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from dataclasses import dataclass
from pathlib import Path

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", 'model.pkl')

class ModelTrainer:
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model training initiate")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], # all rows and all columns except last one which is target column
                train_array[:,-1], # all rows and last column i.e, target column
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Data splitted into independent and dependent features")

            models = {
                "LinearRegression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "RandomForestRegressor": RandomForestRegressor(max_depth=6),
                "Xgboost": XGBRegressor(),
                "ElasticNet": ElasticNet()
            }

            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test,models)
            print(model_report)
            print('\n--------------------------------------------------------\n')
            logging.info(f'Model Report: {model_report}')

            # To get best model
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best model found, Model name: {best_model_name}, R2 score: {best_model_score}")
            print('\n------------------------------------------------------------\n')
            logging.info(f"Best model found, Model name: {best_model}, R2 score: {best_model_score}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

        except Exception as e:
            logging.info("Error occured while creating model trainer object")
            raise CustomeException(e,sys)
        
        
        
    
    
    
  