import os
import sys
import pandas as pd

from src.exception.exception import CustomeException
from src.logger.logging import logging
from src.utils.utils import load_object

class PredictPipeline:

    def __init__(self):
        pass

    def predict(self,features):
        try:
            pre_processor_path = os.path.join("artifacts", 'preprocessor_obj.pkl')
            model_path = os.path.join("artifacts", 'model.pkl')
            
            pre_processor = load_object(pre_processor_path)
            logging.info("Pre-processor object loaded")

            model = load_object(model_path)
            logging.info("Model pickle file loaded")

            scaled_feature = pre_processor.transform(features)
            logging.info("Features are transformed")

            prediction = model.predict(scaled_feature)
            logging.info("Predictions done")

            return prediction

        except Exception as e:
            raise CustomeException(e,sys)
        
class CustomData:

    def __init__(self,
                carat:float,
                depth:float,
                table:float,
                x:float,
                y:float,
                z:float,
                cut:str,
                color:str,
                clarity:str ):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            data = pd.DataFrame(custom_data_input_dict)
            logging.info("Custom data dataframe formed")

            return data
        
        except Exception as e:
            logging.info("error occured while running get_data_as_dataframe from prediction_pipeline.py file")
            raise CustomeException(e,sys)