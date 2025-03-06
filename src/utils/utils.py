# utils is helper file were functions are written inorder to avoid complication in main file

import os
import sys
import pickle
import pandas as pd
import numpy as np

from src.exception.exception import CustomeException
from src.logger.logging import logging

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    """
    This function will any model or any preprocessor object
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("{obj} has been saved to {dir_path}")

    except Exception as e:
        logging.info("Exception occurred while saving object in save_object function in utils.py file")
        raise CustomeException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    
        logging.info("Object is loaded")

    except Exception as e:
        logging.info("Exception occurred while loading object in load_object function in utils.py file")
        raise CustomeException(e,sys)

def evaluate_metrics(true, predict):
    r2 = r2_score(true, predict)
    mae = mean_absolute_error(true, predict)
    rmse = np.sqrt(mean_squared_error(true, predict))
    