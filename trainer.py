import logging
from logging.config import dictConfig

from base_module.gradient_boosting_module import GradientBoostModule
from config import *
from base_module.log_config import LogConfig

dictConfig(LogConfig().dict())
logger = logging.getLogger("prod_logger")

if __name__ == "__main__":
    # create model's instance, train and save the trained model
    gbr_module = GradientBoostModule()
    params = gbr_module.train(DB_HOST_LOCAL, GRID_SEARCH_PARAMS)
    gbr_module.write_model(BASE_MODEL_PATH)
    logger.info(f"Training complete, final params: {params}")
