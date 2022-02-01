import logging
from logging.config import dictConfig
from typing import List

import nltk
from fastapi import FastAPI
from pydantic import BaseModel

from base_module.gradient_boosting_module import GradientBoostModule
from base_module.log_config import LogConfig
from config import *

dictConfig(LogConfig().dict())
logger = logging.getLogger("prod_logger")

app = FastAPI(debug=True)

gbr_module = GradientBoostModule()
healthy = False

class Requests(BaseModel):
    data: List

@app.post("/train")
def train():
    global gbr_module
    train_params = gbr_module.model.get_params()
    # create new instance of the model
    # so the old one is available while the new one is being trained
    gbr_module_new = GradientBoostModule()
    gbr_module_new.train(DB_HOST, train_params)
    gbr_module_new.write_model(MODEL_PATH)
    # update current model
    gbr_module = gbr_module_new
    return {"status": "success"}

@app.get("/healthy")
def healthy():
    return {"healthy": healthy}

@app.post("/predict")
def predict(data: Requests):
    res = gbr_module.predict(data.data)
    logger.info(f"Result: {res}")
    return res

@app.on_event("startup")
def load_model():
    nltk.download('stopwords')
    logger.info("Loading model")
    global healthy
    try:
        # try reading server version of the model first
        gbr_module.read_model(MODEL_PATH)
        healthy = True
        logger.info("Model load complete")
    except Exception as e:
        logger.info(f"Cannot find temporary model {str(e)}")
        logger.info(f"Loading base model")
        logger.info("Model load complete")
        try:
            # otherwise, try loading last model trained locally
            gbr_module.read_model(BASE_MODEL_PATH)
            healthy = True
        except Exception as e:
            logger.error(f"SEVERE: No models found {str(e)}")
            raise

