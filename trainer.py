import logging
from logging.config import dictConfig

from base_module.gradient_boosting_module import GradientBoostModule
from config import *
from base_module.log_config import LogConfig

dictConfig(LogConfig().dict())
logger = logging.getLogger("prod_logger")

gbr_module = GradientBoostModule()


if __name__ == "__main__":
    params = gbr_module.train(GRID_SEARCH_PARAMS)
    gbr_module.write_model(BASE_MODEL_PATH)
    logger.info(f"Training complete, final params: {params}")
