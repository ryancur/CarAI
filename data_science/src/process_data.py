import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno

def initial_eda(data: pd.DataFrame) -> None:
    df = data.copy()
    print("Shape of data:", df.shape)
    print("Columns: ", df.columns)
    print("Column Types: ", df.dtypes)
    print("Null Values Per Column: ", df.isnull().sum())

def odometer_buckets(x):
    if x < 10000:
        return "<10k"
    elif x < 20000:
        return "10k-20k"
    elif x < 30000:
        return "20k-30k"
    elif x < 40000:
        return "30k-40k"
    elif x < 50000:
        return "40k-50k"
    elif x < 60000:
        return "50k-60k"
    elif x < 70000:
        return "60k-70k"
    elif x < 80000:
        return "70k-80k"
    elif x < 90000:
        return "80k-90k"
    elif x < 100000:
        return "90k-100k"
    elif x < 110000:
        return "100k-110k"
    elif x < 120000:
        return "110k-120k"
    elif x < 130000:
        return "120k-130k"
    elif x < 140000:
        return "130k-140k"
    elif x < 150000:
        return "140k-150k"
    else:
        return "150k+"

def convert_year_to_str(x):
    return str(int(x))

def get_first_word(string):
    return string.split(" ")[0]

def remove_car_model(model_name):
    # this is to remove erroneous model names from the data
    # TODO: want to make another one to rename if time permits
    # use with pd apply
    model_remove_list = ["f", "f-", "$362.47", "(cng)", "-", "/", "/vmi", "1/2", "-150", "-benz", "1500/4x4", "40k", "3/4", "5.71", "all", "all-new", "allroad", "alltrack", "armada,platinum", "awd", "big", "dealer*", "fiesta/at", "g.", "grand+cherokee", "golf+gti", "hardtop", "mkx(awd)limited", "mkz,", "mkz/zephyr", "new", "other", "pick", "pickup", "rogue+sport", "si;verado", "silverado/sierra", "sivlerado", "t&c", "t/c", "tow", "transit+connect", "tuscon-", "very"]
    if model_name in model_remove_list:
        return None
    return model_name


if __name__ == "__main__":

    pd.set_option("display.max_columns", 100)

    df = pd.read_csv("../data/raw_data/vehicles.csv")

    # initial check of data
    #initial_eda(data=df)

    # check the unique values in each column
    columns = df.columns.tolist()

    # check the number of unique values per column
    num_unique = df.nunique(axis=0)

    # visualize the missing data
    # msno.matrix(df)
    # plt.show()

    # visualize a heatmap
    # msno.heatmap(df)
    # plt.show()

    # clean the data

    # keep only the last 10 years of data
    df2 = df[df["year"] >= 2010].reset_index(drop=True)

    keep_columns = [
        'price',
        'year',
        'manufacturer',
        'model',
        'paint_color'
    ]

    # drop columns that are not in the keep columns list
    df3 = df2[keep_columns].copy()
    df3 = df3.reset_index(drop=True)

    # drop rows with null values
    df3 = df3.dropna()
    df3 = df3.reset_index(drop=True)

    # check number of unique values
    df3_num_unique = df3.nunique(axis=0)

    # visualize price distribution with histogram
    #df3["price"].hist(bins=10)
    #plt.show()

    # remove outliers - set min price threshold
    df4 = df3[df3["price"] > 1000].copy()
    df4 = df4.reset_index(drop=True)

    # remove outliers - set max price threshold
    # TODO: this can be added to the code above
    df4 = df4[df4["price"] < 300000].copy()
    df4 = df4.reset_index(drop=True)

    # remove duplicates
    df4.drop_duplicates(inplace=True, ignore_index=True)

    # create bins for odometer column then apply to odometer column
    #df4["odometer"] = df4["odometer"].apply(odometer_buckets)

    # convert year to string
    df4["year"] = df4["year"].apply(convert_year_to_str)

    # reduce car model string to one word
    df4["model"] = df4["model"].apply(get_first_word)

    # remove models that are incorrect
    print("Number of rows before removing certain model names: ", df4.shape[0])
    df4["model"] = df4["model"].apply(remove_car_model)
    df4 = df4.dropna()
    df4 = df4.reset_index(drop=True)
    print("Number of rows after removing certain model names: ", df4.shape[0])

    # TODO: combine manufacturer and model bc model depends on manufacturer

    # save to csv
    save_df = df4
    save_df.to_csv("../data/clean_data/processed_data.csv", index=False)

    # get data stats
    num_rows = save_df.shape[0]
    num_cols = save_df.shape[1]
    column_list = save_df.columns.tolist()
    num_uniques = save_df.nunique()

    # print data stats
    print("---------------  DATA STATS  ---------------")
    print("Number of Rows: ", num_rows)
    print("Number of columns: ", num_cols)
    print("Columns: ", column_list)
    print("Number of Uniques Per Column:\n", num_uniques)


    ## FUTURE WORK

    # SMOTE resampling to balance the dataset - test
    # print("Starting SMOTE resampling...")
    # from imblearn.over_sampling import SMOTE
    # sm = SMOTE(random_state=42)
    # X_res, y_res = sm.fit_resample(X, y)
    # print("Finished SMOTE resampling.")
