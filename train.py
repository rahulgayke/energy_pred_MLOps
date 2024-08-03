# This code snippet is to train and save the machine learning models
# The datset was extracted from https://archive.ics.uci.edu/ml/datasets/energy+efficiency
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import joblib
import mlflow
import mlflow.sklearn
import argparse
from sklearn.tree import DecisionTreeRegressor

def load_data(path):
    df = pd.read_csv(path)
    # Naming the columns based on the documentation from machine learning repository (link above)
    for col in df.columns:
        if col=='Y1':
            df.rename(columns={col:'Heating_Load'}, inplace=True)
        if col=='Y2':
            df.rename(columns={col:'Cooling_Load'}, inplace=True)
        if col=='X1':
            df.rename(columns={col:'Relative_compactness'}, inplace=True)
        if col=='X2':
            df.rename(columns={col:'Surface_area'}, inplace=True)
        if col=='X3':
            df.rename(columns={col:'Wall_area'}, inplace=True)
        if col=='X4':
            df.rename(columns={col:'Roof_area'}, inplace=True)
        if col=='X5':
            df.rename(columns={col:'Overall_height'}, inplace=True)
        if col=='X6':
            df.rename(columns={col:'Orientation'}, inplace=True)
        if col=='X7':
            df.rename(columns={col:'Glazing_area'}, inplace=True)
        if col=='X8':
            df.rename(columns={col:'Glazing_area_distribution'}, inplace=True)
    return df

def split_datset(df):
    X = df.drop(["Heating_Load","Cooling_Load"], axis=1)
    y1 = df['Heating_Load']
    y3 = df['Cooling_Load'] # we want to use y3 to geenrate y2

    # X_train will be used for model building and parametrs tunning and X_test is used for testing the model
    X_train, X_test, y_train_1, y_test_1 = train_test_split(X, y1, test_size=0.2, random_state=4)
    y_train_2 = y3[y_train_1.index]
    y_test_2 = y3[y_test_1.index]

    return X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2

def scale_data(X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2):
    # scaling the input data
    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train)
    scaled_X_train = scaler_x.transform(X_train)
    scaled_X_test = scaler_x.transform(X_test)

    # scaling the output data
    scaler_y1 = MinMaxScaler()
    scaler_y1.fit(y_train_1.values.reshape(-1, 1))
    scaled_y_train_1 = scaler_y1.transform(y_train_1.values.reshape(-1, 1))
    scaled_y_train_1 = scaled_y_train_1.ravel()
    scaled_y_test_1 = scaler_y1.transform(y_test_1.values.reshape(-1, 1))
    scaled_y_test_1 = scaled_y_test_1.ravel()

    scaler_y2 = MinMaxScaler()
    scaler_y2.fit(y_train_2.values.reshape(-1, 1))
    scaled_y_train_2 = scaler_y2.transform(y_train_2.values.reshape(-1, 1))
    scaled_y_train_2 = scaled_y_train_2.ravel()
    scaled_y_test_2 = scaler_y2.transform(y_test_2.values.reshape(-1, 1))
    scaled_y_test_2 = scaled_y_test_2.ravel()

    return scaled_X_train, scaled_y_train_1, scaled_y_train_2, scaled_X_test, scaler_y1, scaler_y2, scaler_x

def train_rf_regressor(scaled_X_train, scaled_y_train_1, scaled_y_train_2, params_dict):
    # Heating Load
    n_estimators = params_dict["n_estimators"]
    max_depth = params_dict["max_depth"]
    min_samples_split = params_dict["min_samples_split"]

    rf_1 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split) #n_estimators = 1000, random_state = 42
    # rf_random_1 = RandomizedSearchCV(estimator = rf_1, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
    rf_1.fit(scaled_X_train, scaled_y_train_1)

    # Cooling Load
    rf_2 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split) #n_estimators = 1000, random_state = 42
    # rf_random_2 = RandomizedSearchCV(estimator = rf_2, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
    rf_2.fit(scaled_X_train, scaled_y_train_2)

    return rf_1, rf_2

def eveluate_model(rf_1, scaled_X_test, scaler_y1, rf_2, scaler_y2):
    y_pred1 = rf_1.predict(scaled_X_test)
    y_pred1 = scaler_y1.inverse_transform(y_pred1.reshape(-1, 1))
    y_pred2 = rf_2.predict(scaled_X_test)
    y_pred2 = scaler_y2.inverse_transform(y_pred2.reshape(-1, 1))

    return y_pred1, y_pred2

def dump_scalers(scaler_x, scaler_y1, scaler_y2):
    # scalers
    joblib.dump(scaler_x,"./models/scaler_x.pkl")
    joblib.dump(scaler_y1,"./models/scaler_y1.pkl")
    joblib.dump(scaler_y2,"./models/scaler_y2.pkl")

def dump_models(rf_1, rf_2):
    # models
    joblib.dump(rf_1, "./models/rf_1.joblib")
    joblib.dump(rf_2, "./models/rf_2.joblib")

def main(n_estimators, max_depth, min_samples_split):
    # load the data\
    path = "./data/ENB2012_CSV_data.csv"
    df = load_data(path)

    # train/test split
    X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2 = split_datset(df)

    # Scaling splitted data
    scaled_X_train, scaled_y_train_1, scaled_y_train_2, scaled_X_test, scaler_y1, scaler_y2, scaler_x = scale_data(X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2)

    # Create the random grid
    # random_grid = {
    #     # Number of trees in random forest
    #     "n_estimators" : [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
    #     # Number of features to consider at every split
    #     "max_features" : ['auto', 'sqrt'],
    #     # Maximum number of levels in tree
    #     "max_depth" : [int(x) for x in np.linspace(10, 110, num = 11)], 
    #     # Minimum number of samples required to split a node
    #     "min_samples_split" : [2, 5, 10],
    #     # Minimum number of samples required at each leaf node
    #     "min_samples_leaf" : [1, 2, 4],
    #     # Method of selecting samples for training each tree
    #     "bootstrap" : [True, False],
    # }
    params_dict = {
        "n_estimators" : n_estimators,
        "max_depth" : max_depth,
        "min_samples_split" : min_samples_split,
    }

    # Training the random forest model using RandomizedSearchCV for optimization of parameters
    rf_1, rf_2 = train_rf_regressor(scaled_X_train, scaled_y_train_1, scaled_y_train_2, params_dict)

    # model evaluation
    y_pred1, y_pred2 = eveluate_model(rf_1, scaled_X_test, scaler_y1, rf_2, scaler_y2)
    r2_score_rf1 = r2_score(y_test_1,y_pred1)*100
    r2_score_rf2 = r2_score(y_test_2,y_pred2)*100

    # print the results of evaluation        
    print("#### Heating Load Models were trained and tested with average r2 score of {:0.2f} %".format(r2_score_rf1))
    print("")
    print("#### Cooling Load Models were trained and tested with average r2 score of {:0.2f} %".format(r2_score_rf2))

    # save the models
    dump_scalers(scaler_x, scaler_y1, scaler_y2)
    dump_models(rf_1, rf_2)

    print("Done Model Training")

    # Logging parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split",  min_samples_split)

    # Logging Metric
    # mlflow.log_metric(r2_score_rf1, "r2_score for heating load model")
    # mlflow.log_metric(r2_score_rf2, "r2_score for cooling load model")
    mlflow.log_metric("r2_score for heating load model", r2_score_rf1)
    mlflow.log_metric("r2_score for cooling load model", r2_score_rf2)

    # Logging model
    mlflow.sklearn.log_model(rf_1, "model for heating load")
    mlflow.sklearn.log_model(rf_2, "model for Cooling load")

    print("Completed mlflow loggings")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)

    args = parser.parse_args()

    with mlflow.start_run():
        main(args.n_estimators, args.max_depth, args.min_samples_split)


