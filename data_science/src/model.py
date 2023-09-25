import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import pickle
import json


def rmse(mse: float) -> float:
    """Calculate the Root Mean Squared Error"""
    return np.sqrt(mse)


def read_from_json(file_handle: str) -> dict:
    """Load a json file"""
    with open(file_handle, "r") as open_file:
        json_object = json.load(open_file)
    return json_object


def save_to_json(data: dict, file_handle: str) -> None:
    """Save python dictionary to a json file"""
    with open(file_handle, "w") as out_file:
        json.dump(data, out_file)


MODEL_FILEPATH = "../models/model_file.pkl"
FEATURE_STORE_FILEPATH = "../models/model_feature_store.json"


if __name__ == "__main__":
    pd.set_option("display.max_columns", 50)

    # import data
    df = pd.read_csv("../data/clean_data/processed_data.csv")

    # convert year column from int to str
    df["year"] = df["year"].values.astype(str)

    # drop columns until we get convergence
    # drop_columns = []
    # df.drop(columns=drop_columns, inplace=True)

    # process the data for ML
    # break out price from the rest of the data
    y = df["price"]
    X = df.drop(columns=["price"])

    # get dummies on the categorical data - sparse matrix
    X_sparse_matrix = pd.get_dummies(X, dtype=int)

    ## store features - both sparse matrix and original
    # load feature store
    feature_store_dict = read_from_json(file_handle=FEATURE_STORE_FILEPATH)

    # get features
    X_original_cols = df.columns.tolist()
    X_sparse_cols = X_sparse_matrix.columns.tolist()

    # add features to feature store
    feature_store_dict["original_feature_list"] = X_original_cols
    feature_store_dict["feature_list"] = X_sparse_cols

    # save the feature store
    save_to_json(data=feature_store_dict, file_handle=FEATURE_STORE_FILEPATH)

    # do a train test split on the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sparse_matrix, y, test_size=0.2, random_state=42
    )

    models = {
        "linear_regression": {
            "model": LinearRegression(),
            "train_rmse": None,
            "test_rmse": None,
        },
        "random_forest": {
            "model": RandomForestRegressor(),
            "train_rmse": None,
            "test_rmse": None,
        },
        "svm": {"model": SVR(), "train_rmse": None, "test_rmse": None},
    }

    for model_type, model_data in models.items():
        model = model_data["model"]
        print(f"Creating {model_type} model...")
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        mse_train = mean_squared_error(y_train, preds)
        rmse_train = rmse(mse_train)
        print(f"Training Set {model_type} RMSE: ", rmse_train)
        pred_test = model.predict(X_test)
        mse_test = mean_squared_error(y_test, pred_test)
        rmse_test = rmse(mse_test)
        print(f"Test Set {model_type} RMSE: ", rmse_test)
        models[model_type]["train_rmse"] = rmse_train
        models[model_type]["test_rmse"] = rmse_test

    # compare the test rmse for each model and select the model with the lowest value

    # instantiate the model
    # model_type = "Linear Regression"
    # model = LinearRegression()

    # model_type = "Gaussian Naive Bayes"
    # model = GaussianNB()

    # model_type = "Random Forest"
    # model = RandomForestRegressor()

    # # model_type = "Support Vector Machine"
    # # model = SVR()

    # print(f"Creating {model_type} model...")

    # # train model - fit
    # model.fit(X_train, y_train)

    # # use the model to predict on the training set
    # preds = model.predict(X_train)

    # # evaluate the model on the training set using MSE and RMSE
    # mse_train = mean_squared_error(y_train, preds)
    # rmse_train = rmse(mse_train)
    # print(f"Training Set {model_type} RMSE: ", rmse_train)

    # # evaluate model on the test set
    # pred_test = model.predict(X_test)
    # mse_test = mean_squared_error(y_test, pred_test)
    # rmse_test = rmse(mse_test)
    # print(f"Test Set {model_type} RMSE: ", rmse_test)

    ### PROD MODEL TRAINING ###

    # re-train model for production with all the data
    # prod_model = RandomForestRegressor()
    # prod_model.fit(X_sparse_matrix, y)
    # preds_prod = prod_model.predict(X_sparse_matrix)
    # mse_prod = mean_squared_error(y, preds_prod)
    # rmse_prod = rmse(mse_prod)
    # print(f"Production Model {model_type} RMSE: ", rmse_prod)

    # ### SAVE TRAINED PROD MODEL ###

    # # pickle model
    # print("Pickling the model...")
    # pickle.dump(prod_model, open(MODEL_FILEPATH, "wb"))

    # print("Done.")
