import os
import sys

import numpy as np
import pandas as pd

from src.logger.logging import logging
from src.exception.exception import CustomeException
from src.utils.utils import save_object, evaluate_metrics

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge 
from dataclasses import dataclass
from pathlib import Path

@dataclass 
class ModelTrainerConfig:
    pass

class ModelTrainer:
    
    def __init__(self):
        pass

    def initiate_model_trainer(Self):
        try:
            pass


        except Exception as e:
            raise CustomeException(e,sys)
        
        
        
        pass
    
    
    pass