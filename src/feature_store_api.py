from typing import Optional, List
from dataclasses import dataclass

import hsfs
import hopsworks

import src.config as config
from src.logger import get_logger

logger = get_logger()

@dataclass
class FeatureGroupConfig:
    name: str
    version: int
    description: str
    primary_key: List[str]
    event_time: str
    online_enabled: Optional[bool] = False

@dataclass
class FeatureViewConfig:
    name: str
    version: int
    feature_group: FeatureGroupConfig



def get_feature_store() -> hsfs.feature_store.FeatureStore:
    """Connects to Hopsworks and returns a pointer to the feature store

    Returns:
        hsfs.feature_store.FeatureStore: pointer to the feature store
    """
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )
    return project.get_feature_store()

def get_feature_group(
    name: str,
    version: Optional[int] = 1
    ) -> hsfs.feature_group.FeatureGroup:
    """Connects to the feature store and returns a pointer to the given
    feature group `name`

    Args:
        name (str): name of the feature group
        version (Optional[int], optional): _description_. Defaults to 1.

    Returns:
        hsfs.feature_group.FeatureGroup: pointer to the feature group
    """
    return get_feature_store().get_feature_group(
        name=config.FEATURE_GROUP_METADATA.name,
        version=config.FEATURE_GROUP_METADATA.version,
    )

def get_or_create_feature_group(
    feature_group_metadata: FeatureGroupConfig
) -> hsfs.feature_group.FeatureGroup:
    """Connects to the feature store and returns a pointer to the given
    feature group `name`

    Args:
        name (str): name of the feature group
        version (Optional[int], optional): _description_. Defaults to 1.

    Returns:
        hsfs.feature_group.FeatureGroup: pointer to the feature group
    """
    return get_feature_store().get_or_create_feature_group(
        name=config.FEATURE_GROUP_METADATA.name,
        version=config.FEATURE_GROUP_METADATA.version,
        description=config.FEATURE_GROUP_METADATA.description,
        primary_key=config.FEATURE_GROUP_METADATA.primary_key,
        event_time=config.FEATURE_GROUP_METADATA.event_time,
        online_enabled=config.FEATURE_GROUP_METADATA.online_enabled
    )

def get_or_create_feature_view(
    feature_view_metadata: FeatureViewConfig
) -> hsfs.feature_view.FeatureView:
    """"""

    # get pointer to the feature store
    feature_store = get_feature_store()

    # get pointer to the feature group
    # from src.config import FEATURE_GROUP_METADATA
    feature_group = feature_store.get_feature_group(
        name=config.FEATURE_VIEW_METADATA.feature_group.name,
        version=config.FEATURE_VIEW_METADATA.feature_group.version
    )

    # create feature view if it doesn't exist
    try:
        feature_store.create_feature_view(
            name=config.FEATURE_VIEW_METADATA.name,
            version=config.FEATURE_VIEW_METADATA.version,
            query=feature_group.select_all()
        )
    except:
        logger.info("Feature view already exists, skipping creation.")
    
    # get feature view
    feature_store = get_feature_store()
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_METADATA.name,
        version=config.FEATURE_VIEW_METADATA.version,
    )

    return feature_view