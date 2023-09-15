import pandas as pd
import numpy as np

import pickle
import json

def read_from_json(file_handle: str) -> dict:
    with open(file_handle, "r") as open_file:
        json_object = json.load(open_file)
    return json_object

def combine_model_name(model_name: str) -> str:
    model_name_list = model_name.split(" ")
    return "-".join(model_name_list)


MODEL_FILEPATH = "../models/model_file.pkl"
FEATURE_STORE_FILEPATH = "../models/model_feature_store.json"

""" Automobile Data
[
    {
        "href": "/api/automobiles/1FT7X2B65FEA63551/",
        "id": 3,
        "color": "gray",
        "year": 2020,
        "vin": "1FT7X2B65FEA63551",
        "model": {
            "href": "/api/models/2/",
            "id": 2,
            "name": "F8 Tributo",
            "picture_url": "https://www.focsandiego.com/wp-content/uploads/2021/09/This-Awesome-Ferrari-F8-Tributo-is-for-sale-1024x575.jpg.webp",
            "manufacturer": {
                "href": "/api/manufacturers/3/",
                "id": 3,
                "name": "Ferrari"
            }
        },
        "sold": false
    },
]

NOTE: could create a VO and a poller to get this data and then return "suggested_sale_price"
"""

if __name__ == "__main__":
    pd.set_option('display.max_columns', 50)

    # load the model
    restored_model = pickle.load(open(MODEL_FILEPATH, "rb"))

    # load feature store
    feature_store_dict = read_from_json(file_handle=FEATURE_STORE_FILEPATH)
    feature_list = feature_store_dict.get("feature_list")

    input_df = pd.DataFrame(0, index=np.arange(1), columns=feature_list)

    input_data = {
        "href": "/api/automobiles/1FT7X2B65FEA63551/",
        "id": 3,
        "color": "blue",
        "year": 2015,
        "vin": "1FT7X2B65FEA63551",
        "model": {
            "href": "/api/models/2/",
            "id": 2,
            "name": "Civic",
            "picture_url": "",
            "manufacturer": {
                "href": "",
                "id": 3,
                "name": "Honda"
            }
        },
        "sold": False
        }

    # add sample data to sparse input vector
    new_sample = {
        "year": input_data["year"],
        "manufacturer": input_data["model"]["manufacturer"]["name"].lower(),
        "model": input_data["model"]["name"].lower(),
        "paint_color": input_data["color"].lower()
    }

    # manufacturer_model = f"{new_sample["manufacturer"]}_{new_sample["model"]}" - need helper function
    # manufacturer_model = Ferrari F8 Tributo

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

    print("Sample Data: ", new_dict)
    print("Model Prediction: ", prediction)

    # send a json response or store in database
    price_pred = round(prediction[0], 2)  # 35245.91

    ### view function
    # set up a while loop to check the DB at some time interval (use sleep)

    # get the VO list and the suggestprice list and compare lists

    # if something is in the VO list but not in the suggestedprice list then process
    # the VINs that are not in the suggestedprice list

    # STEP1:  GET vehicle data from http://localhost:8100/api/automobiles/1C3CC5FB2AN120174/
    # for a specific automobile using the vin

    # STEP2: process the data and run it through the model to get a price prediction

    # STEP3: store the prediction to the SuggestedPrice model in the DB

    """
    Considerations:
        - this might be better with a pub/sub or message queue architecture to let the consumer know when a new vin is
            created
    """
