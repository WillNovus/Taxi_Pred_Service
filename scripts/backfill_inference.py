from argparse import ArgumentParser
from pdb import set_trace as stop

import pandas as pd

from src.inference import (
    load_batch_of_features_from_store,
    load_model_from_registry,
    get_model_predictions
)
from src.feature_store_api import get_feature_store
import src.config as config

def run(current_date: pd.Timestamp) -> None:
    """Runs inference onthe given `current_date`
    This function is useful to backfill past inference runs.

    Args:
        current_date (pd.Timestamp): date and time for which we run the batch
        inference
    """
    # step 1. load batch of features from the store
    features = load_batch_of_features_from_store(current_date)
    
    # step 2. load model from the model registry
    model = load_model_from_registry()

    # step 3. generate predictions
    predictions = get_model_predictions(model, features)
    # add a column with the timestamp
    predictions['pickup_hour'] = current_date

    # step 4. save predictions back to the feature store, so we can use them
    # for downstream tasks, like monitoring
    # get a pointer to the feature group
    feature_group = get_feature_store().get_or_create_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTIONS,
        version=1,
        description="Predictions generate by our production model",
        primary_key = ['pickup_location_id', 'pickup_hour'],
        event_time='pickup_hour',
    )
    # save data to the feature group
    feature_group.insert(predictions, write_options={"wait_for_job": False})


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--from_datetime',
                        type=lambda s: pd.Timestamp(s),
                        required=True)
    parser.add_argument('--to_datetime',
                        type=lambda s: pd.Timestamp(s),
                        required=True)
    args = parser.parse_args()

    # list of datetimes we wanna backfill
    date_times_to_backfill = pd.date_range(
        args.from_datetime, args.to_datetime, freq='H')
    
    for current_date in date_times_to_backfill:
        print(current_date)
        run(current_date)
    # stop()
