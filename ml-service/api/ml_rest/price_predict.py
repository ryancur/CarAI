from django.conf import settings

import pandas as pd
import numpy as np

import pickle
import json
import os


def combine_model_name(model_name: str) -> str:
    model_name_list = model_name.split(" ")
    return "-".join(model_name_list)


def open_model(file_name: str, file_type: str) -> any:
    models_folder = settings.BASE_DIR / "ml_rest" / "ml_models"
    file_path = os.path.join(models_folder, os.path.basename(file_name))
    if os.path.exists(file_path) and file_type == "pkl":
        print("Loading Trained Model")
        model = pickle.load(open(file_path, "rb"))
    elif os.path.exists(file_path) and file_type == "json":
        with open(file_path, "r") as open_file:
            model = json.load(open_file)
    else:
        print("No model with this name, check this and retry")
        model = None
    return model


MODEL_FILENAME = "model_file.pkl"
FEATURE_STORE_FILENAME = "model_feature_store.json"

# load the model - move this to dockerfile to load on run
restored_model = open_model(file_name=MODEL_FILENAME, file_type="pkl")

# load feature store - move this to dockerfile to load on run
feature_store_dict = open_model(file_name=FEATURE_STORE_FILENAME, file_type="json")


def predict_price(input_data: dict) -> dict:
    feature_list = feature_store_dict.get("feature_list")

    input_df = pd.DataFrame(0, index=np.arange(1), columns=feature_list)

    # add sample data to sparse input vector
    new_sample = {
        "year": input_data["year"],
        "manufacturer": input_data["manufacturer"].lower(),
        "model": input_data["model"].lower(),
        "paint_color": input_data["color"].lower(),
    }

    new_dict = {}
    for key, value in new_sample.items():
        if key == "model":
            value = combine_model_name(model_name=value)
        feature_name = f"{key}_{str(value)}"
        new_dict[feature_name] = value

    # add a 1 to the columns with the same names as the keys
    for k, v in new_dict.items():
        if k in feature_list:
            input_df.at[0, k] = 1
        else:
            print(f"{k} not in feature list")

    # predict the price
    prediction = restored_model.predict(input_df)

    # send a json response or store in database
    price_pred = round(prediction[0], 2)

    return price_pred
