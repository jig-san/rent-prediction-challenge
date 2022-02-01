import json
import logging
import pickle
import time
from copy import deepcopy
from datetime import timedelta
from logging.config import dictConfig

import pandas as pd
import pandas.io.sql as psql
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sqlalchemy import create_engine

from base_module.log_config import LogConfig
from base_module.module_config import *
from config import *

dictConfig(LogConfig().dict())
logger = logging.getLogger("prod_logger")


def _split_into_train_val_test(dataframe, train_ratio=.8, val_ratio=.2):
    """
    Input: DataFrame and split ratios
    - Split data into 80% train and 20% test
    - Hold out 20% of training data for validation
    Output: three DataFrames with the train / validation / test splits
    """
    train_end = int(train_ratio * len(dataframe))
    val_start = int((1 - val_ratio) * train_end)

    train_set = dataframe[: val_start]
    val_set = dataframe[val_start: train_end]
    test_set = dataframe[train_end:]

    return train_set, val_set, test_set

def _read_df(db_host):
    try:
        logger.info("Reading data from database")
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{db_host}:{DB_PORT}/rentaldata")
        connection = engine.connect()
        return psql.read_sql_table("immo_data", connection).drop("index", axis=1)
    except Exception as e:
        logger.info(f"Table not found: {str(e)}")
        logger.info(f"Fetching local .csv data")
        return pd.read_csv(LOCAL_DATA_PATH)


def _get_non_outlier_indices(series):
    """Filters upper and lower outliers from input Series"""

    q_low = series.quantile(0.01)
    q_hi = series.quantile(0.99)

    return (series < q_hi) & (series > q_low)


def _filter_invalid_targets(dataframe):
    """Removes rows with missing or unrealistic target values from the dataset"""
    # find and remove outliers in the target column
    dataframe = dataframe[_get_non_outlier_indices(dataframe.totalRent)]
    # remove rows where target is NaN or zero
    dataframe = dataframe.drop(dataframe[dataframe.totalRent.isnull()].index)
    dataframe = dataframe.drop(dataframe[dataframe.totalRent == 0].index)
    # reset index
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe

def _write_pickle_object(obj, filename):
    with open(filename, "wb") as out_path:
        pickle.dump(obj, out_path, pickle.HIGHEST_PROTOCOL)

def _read_pickle_object(filename):
    with open(filename, "rb") as in_path:
        return pickle.load(in_path)

def _get_processed_documents(documents):
    """
    Preprocess text (lowercase, tokenize, remove stopwords, stemming)
    :param documents: list of documents
    :return: list of preprocessed documents
    """
    tokenizer = RegexpTokenizer(r'\w+')
    german_stop_words = stopwords.words('german')
    german_stemmer = GermanStemmer()

    documents_list = []

    for i, d in enumerate(documents):
        d_lower = d.lower()
        tokens = tokenizer.tokenize(d_lower)
        stopped = [w for w in tokens if w not in german_stop_words]
        stemmed = " ".join([german_stemmer.stem(w) for w in stopped])

        documents_list.append(stemmed)

    return documents_list

def _get_search_configurations(grid):
    """
    Unfolds grid with lists of parameters to search into list of dicts with model's configurations
    :param grid: dict with string keys (names of model's hyperparameters) and list values (values to search)
    :return: list of dicts with string keys and single values
    """
    grid = deepcopy(grid)

    if not grid:
        return [{}]

    key_first = next(iter(grid))
    values_first = grid[key_first]
    del grid[key_first]

    configs = []
    for val in values_first:
        conf = _get_search_configurations(grid)
        for param in conf:
            param[key_first] = val
            configs.append(param)

    return configs

def _train_with_params(train_X, train_Y, val_X, val_Y, params):
    """
    Fits Gradient Boosting Regression with given hyperparameters
    :return: fit model, train and validation scores
    """
    logger.info("Training in progress ...")
    gbr = GradientBoostingRegressor(**params)
    gbr.fit(train_X, train_Y)
    score_val = gbr.score(val_X, val_Y)
    score_train = gbr.score(train_X, train_Y)
    logger.info(f"Train score: {score_train}")
    logger.info(f"Validation score: {score_val}")

    return gbr, score_train, score_val

def _train_with_grid_search(train_X, train_Y, val_X, val_Y, configurations):
    max_score = 0
    best_model = None
    logger.info("Grid search started")
    train_start = time.time()
    for i, params in enumerate(configurations):
        logger.info(f"Running model {i + 1} out of {len(configurations)}")
        model, score_train, score_val = _train_with_params(train_X, train_Y, val_X, val_Y, params)
        if score_val > max_score:
            max_score = score_val
            best_model = model
            logger.info(f"*** New best score: {max_score}")
            logger.info(f"*** Parameters: {best_model.get_params()}")
    logger.info(f"Finished in {timedelta(seconds=time.time() - train_start)}.")

    return best_model

class GradientBoostModule:
    def __init__(self):
        self.model = None
        self.encoder_tfidf = None
        self.encoder_one_hot = None
        self.encoder_labels = None

    def train(self, db_host, model_params):

        logger.info("Reading training data")

        dataframe = _read_df(db_host)
        dataframe = dataframe.reindex(sorted(dataframe.columns), axis=1)

        logger.info("Reading complete")

        # shuffling the dataset once
        dataframe = shuffle(dataframe, random_state=42)

        # throw away the rows where targets are undefined
        dataframe = _filter_invalid_targets(dataframe)

        # split into target and feature columns
        df_target = dataframe.totalRent

        # the value for USE_TEXT_FIELDS is set in config.py
        if not USE_TEXT_FIELDS:
            for col in TEXT_FIELDS:
                dataframe = dataframe.drop(col, axis=1)

        self.create_encoders(dataframe)

        dataframe = self.process_features(dataframe)

        logger.info("Splitting data into train, val, test")
        train_Y, val_Y, test_Y = _split_into_train_val_test(df_target.values)
        train_X, val_X, test_X = _split_into_train_val_test(dataframe.values)
        logger.info("Splits ready")

        if all(isinstance(v, list) for v in model_params.values()):
            model_configs = _get_search_configurations(model_params)
            self.model = _train_with_grid_search(train_X, train_Y, val_X, val_Y, model_configs)
            logger.info(f"Test score :{self.model.score(test_X, test_Y)}")
        elif any(isinstance(v, list) for v in model_params.values()):
            logger.error("ERROR: wrong format parameters")
        else:
            self.model, score_train, score_val = _train_with_params(train_X, train_Y, val_X, val_Y, model_params)
            logger.info(f"Test score: {self.model.score(test_X, test_Y)}")

        return self.model.get_params()

    def predict(self, records):
        dataframe = pd.DataFrame.from_records(records)
        for col in dataframe.columns:
            if col not in DB_COLUMNS:
                dataframe.drop(col, axis=1, inplace=True)
                logger.info(f"Removed unknown field {col}")
        for col in DB_COLUMNS:
            if col not in dataframe.columns:
                dataframe.insert(0, col, 0)
                logger.error(f"SEVERE: missing field {col}")
                raise ValueError(f"Missing field {col} in data")
        dataframe = dataframe.reindex(sorted(dataframe.columns), axis=1)
        try:
            logger.info(dataframe.columns.to_list())
            feats = self.process_features(dataframe)
            predicted = self.model.predict(feats.values).tolist()
            logger.info(predicted)
            return {"status": "success", "message": json.dumps(predicted)}
        except Exception as e:
            logger.error(f"Error in feature preparation: {str(e)}")
            return {"status": "error", "message": str(e)}

    def write_model(self, path):
        obj = {
            "model": self.model,
            "encoder_tfidf": self.encoder_tfidf,
            "encoder_one_hot": self.encoder_one_hot,
            "encoder_labels": self.encoder_labels
        }
        _write_pickle_object(obj, path)

    def read_model(self, path):
        obj = _read_pickle_object(path)
        self.model, self.encoder_tfidf, self.encoder_one_hot, self.encoder_labels = \
            obj["model"], obj["encoder_tfidf"], obj["encoder_one_hot"], obj["encoder_labels"]

    def process_features(self, dataframe):
        dataframe.drop(FIELDS_TO_IGNORE, axis=1, inplace=True)
        # if the method is called from predict, dataframe does not have target column
        if "totalRent" in dataframe:
            dataframe.drop("totalRent", axis=1, inplace=True)
        dataframe = self.transform_struct_fields(dataframe)
        if USE_TEXT_FIELDS:
            dataframe = self.transform_text_fields(dataframe)
        # fill missing data
        dataframe.fillna(0, inplace=True)
        return dataframe

    def transform_struct_fields(self, dataframe):
        """Prepare structural data for the model"""
        df_columns_list = dataframe.columns.to_list()
        logger.info("Processing structural data")
        for col in df_columns_list:
            # transform selected categorical features into one hot vectors
            if col in ONE_HOT_COLUMNS:
                one_dataframe = pd.DataFrame(self.encoder_one_hot[col].transform(dataframe[col].values.reshape(-1, 1)).toarray())
                dataframe = pd.concat([dataframe, one_dataframe], axis=1).drop(col, axis=1)
            # transform True/False boolean values to 1/0
            elif dataframe[col].dtype == "bool":
                dataframe[col] = dataframe[col].astype("int")
            # encode remaining categorical features as labels
            elif dataframe[col].dtype == "object" and col not in TEXT_FIELDS:
                labels_dataframe = pd.DataFrame(self.encoder_labels[col].transform(dataframe[col].astype(str))).astype("int")
                dataframe[col] = labels_dataframe
            # transform years
            elif col in YEAR_COLUMNS:
                dataframe[col] = dataframe[col].fillna(0).astype("int")
        logger.info("Processing complete")

        return dataframe

    def transform_text_fields(self, dataframe):
        """Prepare text data using TF-IDF with unigrams"""
        logger.info("Processing text data")
        for col in TEXT_FIELDS:
            processed_documents = _get_processed_documents(dataframe[col].astype("str").to_list())
            text_features = self.encoder_tfidf[col].transform(processed_documents)
            dataframe = pd.concat([dataframe, pd.DataFrame(text_features.todense())], axis=1)
        dataframe.drop(list(TEXT_FIELDS), axis=1, inplace=True)
        logger.info("Text data processed")

        return dataframe

    def create_encoders(self, dataframe):
        logger.info("Fitting OneHotEncoder, TfidfVectorizer and LabelEncoder")
        encoders_start = time.time()
        self.encoder_one_hot = {}
        self.encoder_labels = {}
        self.encoder_tfidf = {}
        for col in dataframe.columns.to_list():
            if col in ONE_HOT_COLUMNS:
                encoder = OneHotEncoder(handle_unknown='ignore')
                encoder.fit(dataframe[col].values.reshape(-1, 1))
                self.encoder_one_hot[col] = encoder
            elif col in TEXT_FIELDS:
                processed_documents = _get_processed_documents(dataframe[col].astype("str").to_list())
                vectorizer = TfidfVectorizer(lowercase=False, max_features=1000)
                vectorizer.fit(processed_documents)
                self.encoder_tfidf[col] = vectorizer
            elif dataframe[col].dtype == "object":
                encoder = LabelEncoder()
                encoder.fit(dataframe[col].astype(str))
                self.encoder_labels[col] = encoder
        logger.info(f"Encoders ready in {timedelta(seconds=time.time() - encoders_start)}")
