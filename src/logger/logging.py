import os
import logging

from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #log file name

log_path = os.path.join(os.getcwd(), 'logs') # It will get current working directory and create 'logs' name dir

os.makedirs(log_path, exist_ok=True) # It will create a dir at the path given

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE) # Path of log file to be created

logging.basicConfig(filename = LOG_FILE_PATH,
                    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO
                )