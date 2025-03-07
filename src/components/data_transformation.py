import os
import sys

import numpy as np
import pandas as pd
import pickle

from src.logger.logging import logging
from src.exception.exception import CustomeException
from src.utils.utils import save_object

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from dataclasses import dataclass
from pathlib import Path

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor_obj.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logging.info("Data Transformation initiated")

            # Define which columns to be ordinal and ohe hot encoded
            categorical_columns = ['cut', 'color', 'clarity']
            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define custom ranking for ordinal variable
            cut_cat = ['Fair','Good','Very Good', 'Premium', 'Ideal']
            color_cat = ['D','E' ,'F', 'G',  'H', 'I','J' ]
            clarity_cat = ['I1','SI2','SI1','VS2',  'VS1', 'VVS2', 'VVS1','IF' ]
            
            logging.info("Pipeline Initiated")

            numerical_pipeline = Pipeline(
                steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaling', StandardScaler())
                    ]
                )
            categorical_pipeline = Pipeline(
                steps=[
                        ('cat_impute', SimpleImputer(strategy='most_frequent')),
                        ('or_encoding', OrdinalEncoder(categories=[cut_cat,color_cat,clarity_cat])),
                        ('scaler', StandardScaler())
                    ]
                )
            
            preprocessor = ColumnTransformer(
                    [
                        ('num_pipeline', numerical_pipeline,numerical_columns),
                        ('cat_pipeline',categorical_pipeline,categorical_columns)
                    ]
                )
            logging.info("Preprocessor object has been created")
            return preprocessor


        except Exception as e:
            raise CustomeException(e,sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data transformation initiated")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test file completed")
            logging.info(f"Train df head: \n{train_df.head().to_string()}")
            logging.info(f"Test df head: \n{test_df.head().to_string()}")

            preprocessing_obj = self.get_data_transformation()

            target_column_name = 'price'
            drop_columns = [target_column_name, 'id']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            trarget_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            trarget_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on traning and testing data")
        # np.c_ --> is concatenating them
            train_arr = np.c_[input_feature_train_arr, np.array(trarget_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(trarget_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("Preprocessing pickle file saved")
            logging.info("Data Transformation completed")
            return train_arr, test_arr
                                                   
        except Exception as e:
            logging.info("Error occured while executing initiate_data_transformation function")
            raise CustomeException(e,sys)
        
        
        
    
    
    
