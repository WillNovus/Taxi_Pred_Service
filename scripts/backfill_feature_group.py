import os
from datetime import datetime

from dotenv import load_dotenv
import pandas as pd

from src.data import load_raw_data, transform_raw_data_into_ts_data
from src.config import FEATURE_GROUP_METADATA, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME
from src.feature_store_api import get_or_create_feature_group
from src.logger import get_logger

logger = get_logger()


def get_historical_rides() -> pd.DataFrame:
    """
    Download historical rides from the NYC Taxi dataset
    """
    from_year = 2022
    to_year = datetime.now().year
    print(f'Downloading raw data from {from_year} to {to_year}')

    rides = pd.DataFrame()
    for year in range(from_year, to_year+1):
        
        # download data for the whole year
        rides_one_year = load_raw_data(year)
        
        # append rows
        rides = pd.concat([rides, rides_one_year])

    return rides


def run():

    logger.info('Fetching raw data from data warehouse')
    rides = get_historical_rides()

    logger.info('Transforming raw data into time-series data')
    ts_data = transform_raw_data_into_ts_data(rides)

    # add new column with the timestamp in Unix seconds
    ts_data['pickup_hour'] = pd.to_datetime(ts_data['pickup_hour'], utc=True)    
    ts_data['pickup_ts'] = ts_data['pickup_hour'].astype(int) // 10**6 # Unix milliseconds

    # get a pointer to the feature group we wanna write to
    feature_group = get_or_create_feature_group(FEATURE_GROUP_METADATA)

    # start a job to insert the data into the feature group
    logger.info('Inserting data into feature group...')
    feature_group.insert(ts_data, write_options={"wait_for_job": False})

if __name__ == '__main__':
    run()