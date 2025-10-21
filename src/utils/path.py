from os import environ
import logging

logger = logging.getLogger(__name__)

data_dir = environ.get("EVAPORE_DATA_DIR")

if data_dir is None:
    logger.error("Environment variable 'EVAPORE_DATA_DIR' is not set. Please set it to the base data directory.")
    raise EnvironmentError("Environment variable 'EVAPORE_DATA_DIR' is not set.")