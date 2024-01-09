import os
from dotenv import load_dotenv
from src.paths import PARENT_DIR
from src.feature_store_api import FeatureGroupConfig, FeatureViewConfig

# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR/'.env')

HOPSWORKS_PROJECT_NAME = 'novusx10' 

try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception("Create an .env file on the project root with the HOPSWORKS_API_KEY")


FEATURE_GROUP_METADATA = FeatureGroupConfig(
    name='time_series_hourly_feature_group',
    version=1,
    description='Feature group with hourly time-series data of historical taxi rides',
    primary_key=['pickup_location_id', 'pickup_ts'],
    event_time='pickup_ts',
    online_enabled=True,
)

FEATURE_VIEW_METADATA = FeatureViewConfig(
    name='time_series_hourly_feature_view', 
    version=3,
    feature_group = FEATURE_GROUP_METADATA,
)

FEATURE_GROUP_PREDICTIONS_METADATA = FeatureGroupConfig(
    name='model_predictions_feature_group',
    version=1,
    description="Predictions generate by our production model",
    primary_key = ['pickup_location_id', 'pickup_ts'],
    event_time='pickup_ts',
)

# added for monitoring purposes
FEATURE_VIEW_PREDICTIONS_METADATA =FeatureViewConfig(
    name='model_predictions_feature_view',
    version=1,
    feature_group=FEATURE_GROUP_PREDICTIONS_METADATA,
)

MODEL_NAME = "taxi_demand_predictor_next_hour"
MODEL_VERSION = 1

MONITORING_FV_NAME = 'monitoring_feature_view'
MONITORING_FV_VERSION = 1

# number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28

# number of iterations we want optuna to perform to find the best hyperparameters
N_HYPERPARAMETER_SEARCH_TRIALS = 3

# maximum Mean Absolute Error we allow our production model to have
MAX_MAE = 4.0