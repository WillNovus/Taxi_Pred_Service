from datetime import datetime, timedelta
from argparse import ArgumentParser
from pdb import set_trace as stop

import pandas as pd

from src import config
from src.data import (
    fetch_ride_events_from_data_warehouse,
    transform_raw_data_into_ts_data,
)
from src.feature_store_api import get_feature_group
from src.logger import get_logger

logger = get_logger()

def run(date: datetime):
    """_summary_

    Args:
        date (datetime): _description_

    Returns:
        _type_: _description_
    """
    logger.info('Fetching raw data from data warehouse')
    # fetch raw ride events from the datawarehouse for the last 28 days
    # we fetch the last 28 days, instead of the last hour only, to add redundancy
    # to the feature_pipeline. This way, if the pipeline fails for some reason,
    # we can still re-write data for that missing hour in a later run.
    rides = fetch_ride_events_from_data_warehouse(
        from_date=(date - timedelta(days=28)),
        to_date=date
    )

    logger.info('Transforming raw data into time-series data')
    # transform raw data into time-series data by aggregating rides per
    # pickup location and hour
    ts_data = transform_raw_data_into_ts_data(rides)

    logger.info('Getting pointer to the feature group we wanna save data to')
    # get a pointer to the feature group we wanna write to
    feature_group = get_feature_group(name=config.FEATURE_GROUP_METADATA.name,
                                      version=config.FEATURE_GROUP_METADATA.version)
    
    logger.info('Start job to insert data into feature group')
    # start a job to insert the data into the feature group
    # we wait, to make sure the job is finished before we exit the script, and
    # the inference pipeline can start using the new data
    feature_group.insert(ts_data, write_options={"wait_for_job": True})
    logger.info('Finished job to insert data into feature group')

if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--datetime',
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
                        help='Datetime argument in the format of YYYY-MM-DD HH:MM:SS')
    args = parser.parse_args()

    # if args.datetime was provided, use it as the current_date, otherwise
    # use the current datetime in UTC
    if args.datetime:
        current_date = pd.to_datetime(args.datetime)
    else:
        current_date = pd.to_datetime(datetime.utcnow()).floor('H')
    
    print(f'Running feature pipeline for {current_date=}')
    run(current_date)