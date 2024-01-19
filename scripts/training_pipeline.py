import os
from datetime import date, timedelta
from typing import Tuple, Optional
from pathlib import Path

from comet_ml import Experiment
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
# from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import optuna

from src.data import transform_ts_data_into_features_and_target
from src import config
from src.paths import PARENT_DIR, DATA_CACHE_DIR
from src.config import FEATURE_VIEW_METADATA, N_HYPERPARAMETER_SEARCH_TRIALS
from src.data_split import train_test_split
from src.feature_store_api import get_or_create_feature_view
from src.model_registry_api import push_model_to_registry
from src.model import get_pipeline                                                                 
from src.logger import get_logger
#from src.discord import send_message_to_channel

logger = get_logger()

#load variables from .env file as environment variables.
load_dotenv(PARENT_DIR/'.env')

def fetch_features_and_targets_from_store(
    from_date: pd.Timestamp,
    to_date: pd.Timestamp,
    step_size: int,
) -> pd.DataFrame:
    """
    Fetches time-series data from the store, transforms it into features and
    targets and returns it as a pandas DataFrame.
    """

    #get pointer to feature view
    logger.info('Getting pointer to feature view...')
    feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)

    # generate training data from the feature view
    ts_data, _ = feature_view.training_data(
        description='Time-series hourly taxi rides',
    )
    from_ts = int(from_date.timestamp())
    to_ts = int(to_date.timestamp())
    ts_data = ts_data[ts_data['pickup_ts'].between(from_ts, to_ts)]

    # sort by pickup_location_id and pickup_hour in ascending order
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)

    # drop `pickup_ts` column
    ts_data.drop('pickup_ts', axis=1, inplace=True)

    print(ts_data)
    len(ts_data)

    # transform time-series data from the feature store into features and targets
    # for supervised learning
    features, targets = transform_ts_data_into_features_and_target(
        ts_data,
        input_seq_len=config.N_FEATURES, # one month
        step_size=step_size,
    )

    features_and_target = features.copy()
    features_and_target['target_rides_next_hour'] = targets

    return features_and_target

def split_data(
    features_and_target: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    # breakpoint()
    X_train, y_train, X_test, y_test = train_test_split(
        features_and_target,
        cutoff_date,
        target_column_name='target_rides_next_hour'   
    )
    logger.info(f'{X_train.shape=}')
    logger.info(f'{y_train.shape=}')
    logger.info(f'{X_test.shape=}')
    logger.info(f'{y_test.shape=}')
    
    return X_train, y_train, X_test, y_test

def find_best_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: Optional[int] = 10,    
) -> dict:
    """"""
    
    def objective(trial: optuna.trial.Trial) -> float:
        """
        Given a set of hyper-parameters, it trains a model and computes an average
        validation error based on a TimeSeriesSplit
        """
        # pick hyper-parameters
        hyperparams = {
            "metric": 'mae',
            "verbose": -1,
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 100),   
        }
        
        tss = TimeSeriesSplit(n_splits=2)
        scores = []
        for train_index, val_index in tss.split(X_train):

            # split data for training and validation
            X_train_, X_val_ = X_train.iloc[train_index, :], X_train.iloc[val_index,:]
            y_train_, y_val_ = y_train.iloc[train_index], y_train.iloc[val_index]
            
            # train the model
            pipeline = get_pipeline(**hyperparams)
            pipeline.fit(X_train_, y_train_)
            
            # evaluate the model
            y_pred = pipeline.predict(X_val_)
            mae = mean_absolute_error(y_val_, y_pred)

            scores.append(mae)
    
        # Return the mean score
        return np.array(scores).mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    logger.info(f'{best_params=}')

    return best_params

def load_features_and_targets(
        local_path_features_and_target: Optional[Path] = None,
    ) -> pd.DataFrame:

        if local_path_features_and_target:
            logger.info('Loading features_and_target from local file')
            features_and_target = pd.read_parquet(local_path_features_and_target)
        else:
            logger.info('Fetching features and targets from the feature store.')
            from_date = pd.to_datetime(date.today() - timedelta(days=52*7))
            to_date = pd.to_datetime(date.today())
            features_and_target = fetch_features_and_targets_from_store(from_date, to_date, step_size = 23)

            try:
                local_file = DATA_CACHE_DIR / 'features_and_target.parquet'
                features_and_target.to_parquet(local_file)
                logger.info(f'Saved features_and_target to local file at {local_file}')
            except:
                logger.info('Could not save features_and_target to local file')
                pass
        return features_and_target 

def train(local_path_features_and_target: Optional[Path]= None) -> None:
    """
    Trains model and pushes it to the model registry if it meets the minimum
    performance threshold.
    """
    logger.info(' Start model training...')

    # Start Comet ML experiment run
    logger.info('Creating Comet ML experiment')
    experiment = Experiment(
        api_key = os.environ["COMET_ML_API_KEY"],
        workspace = os.environ["COMET_ML_WORKSPACE"],
        project_name = os.environ["COMET_ML_PROJECT_NAME"]
    )

    # load features and targets
    features_and_target = load_features_and_targets(local_path_features_and_target)
    experiment.log_dataset_hash(features_and_target)

    # split the data into training and validation sets
    cutoff_date = pd.to_datetime(date.today() - timedelta(days=28), utc=True)
    logger.info('Splitting data into training and test sets...')
    X_train, y_train, X_test, y_test = split_data(
        features_and_target,
        cutoff_date=cutoff_date
    )

    experiment.log_parameters({
        'X_train_shape': X_train.shape,
        'y_train_shape': y_train.shape,
        'X_test_shape': X_test.shape,
        'y_test_shape': y_test.shape,
    })

    # find the best hyperparameters using time-based cross-validation
    logger.info('Finding best hyperparameters...')
    best_hyperparameters = find_best_hyperparameters(X_train, y_train, n_trials=N_HYPERPARAMETER_SEARCH_TRIALS)

    # train the model using the best hyperparameters
    logger.info('Training model using the best hyperparameters...')
    pipeline = get_pipeline(**best_hyperparameters)
    pipeline.fit(X_train, y_train)

    #evaluate the model on test data
    predictions = pipeline.predict(X_test)
    test_mae = mean_absolute_error(y_test, predictions)
    logger.info(f'{test_mae=:.4f}')

    # push the model to the model registry if it meets the minimum performance threshold
    if test_mae < config.MAX_MAE:
        logger.info('Pushing model to the model registry...')
        model_version = push_model_to_registry(
            pipeline,
            X_train_sample=X_train.head(10),
            y_train_sample=y_train.head(10),
            test_mae=test_mae,
            description="LightGBM regressor with a bit of hyper-parameter tuning",
        )
        print(f'New model pushed to the model registry. {test_mae=:.4f}, {model_version=}')

    else:
        logger.info('Model did not meet the minimum performance threshold. Skip pushing to the model registry.')

if __name__ == '__main__':

    from fire import Fire

    Fire(train)