from typing import Optional
from pathlib import Path

import hopsworks
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

import src.config as config
from src.paths import MODELS_DIR
from src.logger import get_logger

logger = get_logger()

def get_model_registry() -> None:
    """Connects to Hopsworks and returns a pointer to the feature store

    Returns:
        hsfs.feature_store.FeatureStore: pointer to the feature store
    """
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    return project.get_model_registry()

def push_model_to_registry(
    model: Pipeline,
    X_train_sample: pd.DataFrame,
    y_train_sample: pd.DataFrame,
    test_mae: float,
    description: Optional[str] = '',
) -> int:
    """"""
    # save the model to disk
    joblib.dump(model, MODELS_DIR / 'model.pkl')

    # define the input and output schema
    input_schema = Schema(X_train_sample)
    output_schema = Schema(y_train_sample)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    # push model to the registry
    model_registry = get_model_registry()
    model = model_registry.sklearn.create_model(
        name="taxi_demand_predictor_next_hour",
        metrics={"test_mae": test_mae},
        description=description,
        input_example=X_train_sample,
        model_schema=model_schema
    )
    model.save(MODELS_DIR / 'model.pkl')

    return model.version

def get_latest_model_from_registry() -> Pipeline:
    """Returns the latest model from the registry"""
    model_registry = get_model_registry()
    model = model_registry.get_models(name=config.MODEL_NAME)[-1]
    logger.info(f'Loading model version {model.version}')

    model_dir = model.download()
    model = joblib.load(Path(model_dir)  / 'model.pkl')

    return model