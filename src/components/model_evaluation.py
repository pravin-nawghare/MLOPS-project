import os
import sys

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import pickle

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
        pass

    def initiate_model_evaluation(Self):
        try:
            pass


        except Exception as e:
            raise CustomeException(e,sys)
        
        
        
        pass
    
    
    pass