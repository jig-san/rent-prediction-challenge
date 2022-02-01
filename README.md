## Service description
This project implements a system for local and server training and predictions on the dataset of with the rental
property data. The main files are:

- `trainer.py ` implements the part which is responsible for the local model training, by default it takes grid search
  parameters as an input.
- `server.py` is a server with three endpoints: **/healthy**, **/train** and **/predict**.
- `base_module/gradient_boosting_module.py` is the server which implements all the ML functionality, including data
  preparation and model training with grid search or with fixed parameters. Both `trainer.py` and `server.py` inherit
  their methods from this module.

## Running the project
The service can be started by running these commands from the project root:

```console
docker-compose build
docker-compose up
```

After this, go to http://localhost:8080/docs to use the Swagger UI.

This sets up and starts two containers: `rent-predict-service` for the server and `rent-predict-db` for the Postgres
database.

**Note:** on the very first startup, the database is filled with the values from the local `immo_data.csv` file, which
should be placed into the root folder of the project.

### Local Training
When the database container is up, you can also start the local training by simply running:

```
python3 trainer.py
```

Local trainer also supports reading from the local dataset file, so in case it cannot connect to the database, place
the `immo_data.csv` file into the root folder, and the trainer will read from it instead.

The trainer writes the trained model to `gbr_model_best.pkl`.

### Server
The server is run on FastApi. This choice was made for the demo purposes, since it provides Swagger UI, which is
convenient in case you want to call the service manually.

The server has three methods:

- `/healthy`: GET method, without the input parameters, that returns the flag indicating if the service is alive. This
  is done for continuous integration purposes. The precise name and returned data has to be adjusted depending on what
  platform is used.
- `/train`: POST method, without any input parameters, which trains the model on the server, using the parameters of the
  current loaded model, and writes it to `gbr_model_best_tmp.pkl`.
- `/predict`: POST method, takes the list of records as input and checks its validity (removes previously unseen fields,
  and errs in case of missing field), and returns the list of predictions obtained with the currently loaded model. The
  format of the input:

```
{
  "data": [
    # place your records here
  ]
}
```

#### Continuous deployment

- **Before every deployment**, the model must be trained locally and the `gbr_model_best.pkl` file must be placed to the
  folder with the server, so that everytime the server is deployed, but its method `/train` was not yet called, it still
  has the available model to do `/predict`.
- **Every time the new app files are deployed**, the old `gbr_model_best_tmp.pkl` file is removed from the server.
- **On every startup** the server creates the instance of the `base_module` and tries to load `gbr_model_best_tmp.pkl`,
  if this model is not found, it tries to load the `gbr_model_best.pkl` instead. This way it always has the most recent
  version of the model.
- When `/train` is called, it uses the loaded model's parameters and creates new base model's instance to train, so
  there's a model available for `/predict`, while the training is in progress.
- If the server manages to load any model, it sets the `healthy` flag to True. Otherwise, it
  raises an exception and exits.

### Base Module

This module is responsible for the functionality of entire system. The libraries I used for the training and feature
processing are Pandas, Sklearn, Numpy, NLTK.

- **Model:** I chose the Gradient Boosting Regression from Sklearn as the base model, since it's a generally
  well-performing model, which is robust and requires minimal data preparation.
- **Feature preparation:** Due to the time limit, and also because I wanted to focus on building more or less
  self-contained system, instead of focusing on the data analysis part of this task, the feature preparation is quite basic.
  The main preprocessing steps are:
    - Remove the rows where the target value **totalRent** is NaN or 0.
    - Remove the rows with the outliers in the target (lower and upper 0.01 quantiles), because I've noticed obvious
      errors in the data, such as the real targets multiplied by 100 compared to their real values, based on Gradient
      Boosting's predictions on that input.
    - Drop the fields as mentioned in the email.
    - And in addition drop the fields with 70% or more missing data, which are **telekomHybridUploadSpeed**,
      **electricityBasePrice**, **electricityKwhPrice**.
    - Drop **streetPlain** and **regio3** because of too many different values.
    - Transform categorical features to one-hot vectors, and some categorical features, where ordering seems to make
      sense, to labels (OneHotEncoder, LabelEncoder from Sklearn).
    - Prepare text fields **description** and **facilities** with lowercase, tokenize, remove stopwords, stemming.
    - Transform prepared text with TfidfVectorizer with unigrams, using 1000 most common features.
    - Fill all the missing values with 0s.

#### Training and results

I trained Gradient Boosting Regressor using 1. Only structural data, 2. Structural data and text data.

The dataset is split into **train** and **test** sets in a 80% to 20% ratio, and 20% of the train split I reserve as
the **validation** set. During the grid search, I evaluate the model trained on the train set on the validation set. The
best model is the one with the highest validation score. Finally, I evaluate the best model on the test set.

As the performance evaluation metric I used coefficient of
determination (<img src="https://render.githubusercontent.com/render/math?math=R^2">), since it is easy to interpret as
the % of the data that fits the regression model.

1. For the training with structural data, I did the grid search, the parameters for which are set in the `config.py`.
   The training of all the 24 resulting models, together with the feature preparation took ~1:35 hours.
2. For training with text, I did the grid search on only 5% of the data (~1:30 hours), and then trained the resulting
   model with the best found parameters (~1:10 hours).
3. In the current version of the app, the training and the parameters search are done without text fields, the flag for
   this option can be changed in `config.py` (USE_TEXT_FIELDS).

I used squared error as the loss function, since small tests I did in advance showed, that this function gives the best
performance compared to the other ones available.

The <img src="https://render.githubusercontent.com/render/math?math=R^2"> scores for each data split of the best models
for both datasets are as follows:

| Dataset         | Train | Val   | Test |
|-----------------|-------|-------|------|
| All Data        | 0.932 | 0.912 |0.913 | 
| Text Excluded   |0.929  |0.915  | 0.916|

The results are quite high, except, as you can see, the text descriptions do not bring any additional value: the scores
for the two models are similar.

### Trained Models

The final trained models live in the `models` folder. I pickle the Gradient Boosting model together with the fitted
feature encoders, so the format of the pickled object is:

```
 {
   "model": GradientBoostingRegressor(),
   "encoder_tfidf": {
      "description": "TfidfVectorizer(),
      "facilities": "TfidfVectorizer()
   },
   "encoder_one_hot": {
      "firingTypes": "OneHotEncoder()",
      ...
   },
   "encoder_labels": {
      "energyEfficiencyClass": "LabelEncoder()",
      ...
    }
 }
```

### Improvements

- As for the problem formulation, and for the evaluation purposes, I train the model on only 64% of the entire dataset
  (because of the data splits). For the production, the code should be adapted, so that the best models are trained on
  the entire dataset.
- The text fields now do not improve the model's performance, which means that it would probably make sense to try
  implementing more careful and probably more sophisticated preparation of the text features.
- The method for dealing with the missing values could be better: NaNs are not always zeroes, so there's also an option
to interpolate them using some smart method instead. 
