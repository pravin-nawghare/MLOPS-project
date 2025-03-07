import os
import sys

import numpy as np
import pandas as pd

from src.logger.logging import logging# if logging is written in init file in logger dir then second logging not required
from src.exception.exception import CustomeException

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass 
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    
    def __init__(self):
        """
        Upper classes variable can be assess from self method all three
        """
        self.ingestion_config = DataIngestionConfig() 

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            data = pd.read_csv("https://github.com/sunnysavita10/Gemstone-Price-Prediction-End-to-End-Pipeline/raw/main/artifacts/raw.csv")
            logging.info("Reading the dataframe")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved as csv file")

            logging.info("Starting train test split operation on raw data")
            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("Data is divided into training and testing data")

            logging.info("saving the data as a csv file...")
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Raw data is saved as train data and test in the artifact directory")

            logging.info("Data ingestion step completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Error occured while executing initiate_data_ingestion function from data_ingestion.py file")
            raise CustomeException(e,sys)
        
        
        

    
    
