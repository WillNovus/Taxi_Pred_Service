import os
from dotenv import load_dotenv
from src.paths import PARENT_DIR

# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR/'.env')

HOPSWORK_PROJECT_NAME = 'taxi_batch_scoring' 

try:
    HOPSWORK_API_KEY = os.environ['HOPSWORK_API_KEY']
except:
    raise Exception("Create an .env file on the project root with the HOPSWORKS_API_KEY")

FEATURE_GROUP_NAME = "time_series_hourly_feature_group"
FEATURE_GROUP_VERSION = 1  
