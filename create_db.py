import logging
from logging.config import dictConfig
from base_module.log_config import LogConfig
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd

from config import *

dictConfig(LogConfig().dict())
logger = logging.getLogger("prod_logger")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/rentaldata")
if not database_exists(engine.url):
    logger.info("Creating database")
    create_database(engine.url)

logger.info("Exporing dataset to database...")
df = pd.read_csv(LOCAL_DATA_PATH)
df.to_sql('immo_data', engine)
logger.info("Export complete")
