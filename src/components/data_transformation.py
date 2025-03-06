import os
import sys

import numpy as np
import pandas as pd

from src.logger.logging import logging
from src.exception.exception import CustomeException

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from dataclasses import dataclass
from pathlib import Path

@dataclass 
class DataTransformationConfig:
    pass

class DataTransformation:
    
    def __init__(self):
        pass

    def initiate_data_transformation(Self):
        try:
            pass


        except Exception as e:
            raise CustomeException(e,sys)
        
        
        
        pass
    
    
    pass