from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import fire

from src.inference import (
    load_batch_of_features_from_store,
    load_model_from_registry,
    get_model_predictions
)
from src.feature_store_api import get_feature_store
from src import config
from src.model_registry_api import get_latest_model_from_registry
from src.logger import get_logger

logger = get_logger()

def inference(
    current_date: Optional[pd.Timestamp] = pd.to_datetime(datetime.utcnow()).floor('H') - timedelta(days=365),
) -> None:
    """"""
    logger.info(f'Running inference pipeline for {current_date}')

    logger.info('Loading batch of features from the feature store')
    features = load_batch_of_features_from_store(current_date)

    logger.info('Loading model from the model registry')
    model = get_latest_model_from_registry()

    logger.info('Generating predictions')
    predictions = get_model_predictions(model, features)
    predictions['pickup_hour'] = current_date

    logger.info('Saving predictions to the feature store')
    feature_group = get_feature_store().get_or_create_feature_group(
        name=config.FEATURE_GROUP_PREDICTIONS_METADATA.name,
        version=1,
        description="Predictions generate by our production model",
        primary_key = ['pickup_location_id', 'pickup_hour'],
        event_time='pickup_hour',
    )
    
    feature_group.insert(predictions, write_options={"wait_for_job": False})
    
    logger.info('Inference DONE!')

if __name__ == '__main__':

    fire.Fire(inference)