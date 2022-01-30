import logging
from logging.config import dictConfig
from typing import List

import nltk
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from base_module.gradient_boosting_module import GradientBoostModule
from config import *
from base_module.log_config import LogConfig

dictConfig(LogConfig().dict())
logger = logging.getLogger("prod_logger")

app = FastAPI(debug=True)

gbr_module = GradientBoostModule()
healthy = False

class Requests(BaseModel):
    data: List

@app.get("/train")
def train():
    train_params = gbr_module.model.get_params()
    gbr_module.train(DB_HOST, train_params)
    gbr_module.write_model(MODEL_PATH)
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
        logger.error(f"Cannot find trained model {str(e)}")
        logger.info(f"Loading base model")
        logger.info("Model load complete")
        try:
            # otherwise, try loading last model trained locally
            gbr_module.read_model(BASE_MODEL_PATH)
            healthy = True
        except Exception as e:
            logger.error(f"SEVERE: No models found {str(e)}")
            raise


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=5000, use_colors=True, reload=True)
