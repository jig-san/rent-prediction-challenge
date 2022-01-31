GRID_SEARCH_PARAMS = {
    "learning_rate": [0.07, 0.01],
    "n_estimators": [150, 300],
    "subsample": [0.5, 0.8, 1.0],
    "max_depth": [3, 6],
    "min_samples_split": [10],
    "min_samples_leaf": [6],
    "random_state": [42]
}
MODEL_PATH = "gbr_model_best_tmp.pkl"
BASE_MODEL_PATH = "gbr_model_best.pkl"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_USER = "postgres"
DB_PASSWORD = "adminpsql"
LOCAL_DATA_PATH = "database/immo_data.csv"
