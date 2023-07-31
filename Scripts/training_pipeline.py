from datetime import date, timedelta
from typing import Tuple, Optional

import hsfs
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
# from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import optuna

from src.data import transform_ts_data_into_features_and_target
from src import config
from src.data_split import train_test_split
from src.feature_store_api import get_feature_store
from src.model_registry_api import push_model_to_registry
from src.model import get_pipeline                                                                  
from src.logger import get_logger
#from src.discord import send_message_to_channel

logger = get_logger()

def get_pointer_to_feature_view() -> hsfs.feature_view.FeatureView:
    """Returns a pointer to the feature view"""
    
    # pointer to feature store
    feature_store = get_feature_store()

    # pointer to feature group
    feature_group = feature_store.get_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION
    )

    try:
        # create feature view if it doesn't exist yet
        feature_store.create_feature_view(
            name=config.FEATURE_VIEW_NAME,
            version=config.FEATURE_VIEW_VERSION,
            query=feature_group.select_all()
        )
    except:
        print('Feature view already existed. Skip creation.')


    # get pointer to the feature view
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )

    # and finally return it
    return feature_view

def get_features_and_targets(
    from_date: str,
    to_date: str,
    step_size: int,
) -> pd.DataFrame:
    """
    Fetches time-series data from the store, transforms it into features and
    targets and returns it as a pandas DataFrame.
    """
    feature_view = get_pointer_to_feature_view()

    # generate training data from the feature view
    ts_data, _ = feature_view.training_data(
        description='Time-series hourly taxi rides',
        start_time=from_date,
        end_time=to_date,
    )

    # sort by pickup_location_id and pickup_hour in ascending order
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)

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


def train() -> None:
    """
    Trains model and pushes it to the model registry if it meets the minimum
    performance threshold.
    """
    logger.info('Training model...')

    # get features and targets from the feature store
    logger.info('Getting features and targets from the feature store...')
    from_date = str(date.today() - timedelta(days=26*7)) # half a year of data
    to_date = str(date.today())
    features_and_target = get_features_and_targets(from_date,to_date, step_size=23)

    # split the data into training and validation sets
    logger.info('Splitting data into training and test sets...')
    X_train, y_train, X_test, y_test = split_data(
        features_and_target,
        cutoff_date=pd.to_datetime(date.today() - timedelta(days=28))
    )

    # find the best hyperparameters using time-based cross-validation
    logger.info('Finding best hyperparameters...')
    best_hyperparameters = find_best_hyperparameters(X_train, y_train, n_trials=1)

    # train the model using the best hyperparameters
    logger.info('Training model using the best hyperparameters...')
    pipeline = get_pipeline(**best_hyperparameters)
    pipeline.fit(X_train, y_train)

    # evalute the model on test data
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
        send_message_to_channel(f'New model pushed to the model registry. {test_mae=:.4f}, {model_version=}')

    else:
        logger.info('Model did not meet the minimum performance threshold. Skip pushing to the model registry.')

if __name__ == '__main__':

    train()