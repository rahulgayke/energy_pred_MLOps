# This code snippet is to train and save the machine learning models
# The datset was extracted from https://archive.ics.uci.edu/ml/datasets/energy+efficiency
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import mlflow
import mlflow.sklearn
import argparse

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

def train_dt_regressor(scaled_X_train, scaled_y_train_1, scaled_y_train_2, params_dict):
    max_depth = params_dict["max_depth"]
    min_samples_split = params_dict["min_samples_split"]

    # Heating Load
    dt_1 = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=33)
    # rf_random_1 = RandomizedSearchCV(estimator = rf_1, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
    dt_1.fit(scaled_X_train, scaled_y_train_1)

    # Cooling Load
    dt_2 = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=44)
    # rf_random_2 = RandomizedSearchCV(estimator = rf_2, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
    dt_2.fit(scaled_X_train, scaled_y_train_2)

    return dt_1, dt_2

def train_rf_regressor(scaled_X_train, scaled_y_train_1, scaled_y_train_2, params_dict):
    n_estimators = params_dict["n_estimators"]
    max_depth = params_dict["max_depth"]
    min_samples_split = params_dict["min_samples_split"]

    # Heating Load
    rf_1 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=11) #n_estimators = 1000, random_state = 42

    rf_1.fit(scaled_X_train, scaled_y_train_1)

    # Cooling Load
    rf_2 = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=22) #n_estimators = 1000, random_state = 42
    # rf_random_2 = RandomizedSearchCV(estimator = rf_2, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
    rf_2.fit(scaled_X_train, scaled_y_train_2)

    return rf_1, rf_2

def train_svm_regressor(scaled_X_train, scaled_y_train_1, scaled_y_train_2, params_dict):
    kernel = params_dict["kernel"]
    c_val = params_dict["c_val"]
    gamma = params_dict["gamma"]
    epsilon = params_dict["epsilon"]

    # Heating Load
    svr_1 = SVR(kernel=kernel, C=c_val, gamma=gamma, epsilon=epsilon)
    # rf_random_1 = RandomizedSearchCV(estimator = rf_1, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
    svr_1.fit(scaled_X_train, scaled_y_train_1)

    # Cooling Load
    svr_2 = SVR(kernel=kernel, C=c_val, gamma=gamma, epsilon=epsilon)
    # rf_random_2 = RandomizedSearchCV(estimator = rf_2, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
    svr_2.fit(scaled_X_train, scaled_y_train_2)

    return svr_1, svr_2

def eveluate_model(heating_model, scaled_X_test, scaler_y1, cooling_model, scaler_y2):
    y_pred1 = heating_model.predict(scaled_X_test)
    y_pred1 = scaler_y1.inverse_transform(y_pred1.reshape(-1, 1))
    y_pred2 = cooling_model.predict(scaled_X_test)
    y_pred2 = scaler_y2.inverse_transform(y_pred2.reshape(-1, 1))

    return y_pred1, y_pred2

def dump_scalers(scaler_x, scaler_y1, scaler_y2):
    # scalers
    joblib.dump(scaler_x,"./models/scaler_x.pkl")
    joblib.dump(scaler_y1,"./models/scaler_y1.pkl")
    joblib.dump(scaler_y2,"./models/scaler_y2.pkl")

def dump_models(heating_model, cooling_model, path_1, path_2):
    # models
    joblib.dump(heating_model, path_1)
    joblib.dump(cooling_model, path_2)

############################################################################################################################
# # DECISION TREE
# def main(max_depth, min_samples_split):
#     # load the data\
#     path = "./data/ENB2012_CSV_data.csv"
#     df = load_data(path)

#     # train/test split
#     X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2 = split_datset(df)

#     # Scaling splitted data
#     scaled_X_train, scaled_y_train_1, scaled_y_train_2, scaled_X_test, scaler_y1, scaler_y2, scaler_x = scale_data(X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2)

#     # Creating parameter dictionary for DT
#     params_dict_dt = {
#         "max_depth" : max_depth,
#         "min_samples_split" : min_samples_split,
#     }

#     # Training the decision tree model
#     dt_1, dt_2 = train_dt_regressor(scaled_X_train, scaled_y_train_1, scaled_y_train_2, params_dict_dt)
#     # model evaluation
#     y_pred1, y_pred2 = eveluate_model(dt_1, scaled_X_test, scaler_y1, dt_2, scaler_y2)
#     r2_score_dt1 = r2_score(y_test_1, y_pred1)*100
#     r2_score_dt2 = r2_score(y_test_2, y_pred2)*100
#     # print the results of evaluation        
#     print("#### DT Heating Load Models were trained and tested with average r2 score of {:0.2f} %".format(r2_score_dt1))
#     print("#### DT Cooling Load Models were trained and tested with average r2 score of {:0.2f} %".format(r2_score_dt2))

#     print("Done DT Model Training")

#     # save the models
#     dump_scalers(scaler_x, scaler_y1, scaler_y2)

#     # Decision Tree Model
#     dt_path_1 = "./models/dt_1.joblib"
#     dt_path_2 = "./models/dt_2.joblib"
#     dump_models(dt_1, dt_2, dt_path_1, dt_path_2)

#     # Decision Tree
#     # Logging parameters
#     mlflow.log_param("dt_max_depth", params_dict_dt["max_depth"])
#     mlflow.log_param("dt_min_samples_split",  params_dict_dt["min_samples_split"])
#     # Logging Metric
#     mlflow.log_metric("r2_score for DT heating load model", r2_score_dt1)
#     mlflow.log_metric("r2_score for DT cooling load model", r2_score_dt2)
#     # Logging model
#     mlflow.sklearn.log_model(dt_1, "model for DT heating load")
#     mlflow.sklearn.log_model(dt_2, "model for DT Cooling load")

#     print("Completed mlflow loggings for DT")

############################################################################################################################
# RANDOM FOREST
def main(n_estimators, max_depth, min_samples_split):
    # load the data\
    path = "./data/ENB2012_CSV_data.csv"
    df = load_data(path)

    # train/test split
    X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2 = split_datset(df)

    # Scaling splitted data
    scaled_X_train, scaled_y_train_1, scaled_y_train_2, scaled_X_test, scaler_y1, scaler_y2, scaler_x = scale_data(X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2)

    # Creating parameter dictionary for RF
    params_dict_rf = {
        "n_estimators" : n_estimators,
        "max_depth" : max_depth,
        "min_samples_split" : min_samples_split,
    }

    # Training the random forest model
    rf_1, rf_2 = train_rf_regressor(scaled_X_train, scaled_y_train_1, scaled_y_train_2, params_dict_rf)
    # model evaluation
    y_pred1, y_pred2 = eveluate_model(rf_1, scaled_X_test, scaler_y1, rf_2, scaler_y2)
    r2_score_rf1 = r2_score(y_test_1, y_pred1)*100
    r2_score_rf2 = r2_score(y_test_2, y_pred2)*100
    # print the results of evaluation        
    print("#### RF Heating Load Models were trained and tested with average r2 score of {:0.2f} %".format(r2_score_rf1))
    print("#### RF Cooling Load Models were trained and tested with average r2 score of {:0.2f} %".format(r2_score_rf2))

    print("Done RF Model Training")

    # # save the models
    # dump_scalers(scaler_x, scaler_y1, scaler_y2)

    # Random Forest Model
    rf_path_1 = "./models/rf_1.joblib"
    rf_path_2 = "./models/rf_2.joblib"
    dump_models(rf_1, rf_2, rf_path_1, rf_path_2)

    # Random Forest
    # Logging parameters
    mlflow.log_param("RF_n_estimators", params_dict_rf["n_estimators"])
    mlflow.log_param("RF_max_depth", params_dict_rf["max_depth"])
    mlflow.log_param("RF_min_samples_split", params_dict_rf["min_samples_split"])
    # Logging Metric
    mlflow.log_metric("r2_score for RF heating load model", r2_score_rf1)
    mlflow.log_metric("r2_score for RF cooling load model", r2_score_rf2)
    # Logging model
    mlflow.sklearn.log_model(rf_1, "model for RF heating load")
    mlflow.sklearn.log_model(rf_2, "model for RF Cooling load")

    print("Completed mlflow loggings for RF")

# ############################################################################################################################
# # SUPPORT VECTOIR MACHINE
# def main(kernel, c_val, gamma, epsilon):
#     # load the data\
#     path = "./data/ENB2012_CSV_data.csv"
#     df = load_data(path)

#     # train/test split
#     X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2 = split_datset(df)

#     # Scaling splitted data
#     scaled_X_train, scaled_y_train_1, scaled_y_train_2, scaled_X_test, scaler_y1, scaler_y2, scaler_x = scale_data(X_train, X_test, y_train_1, y_test_1, y_train_2, y_test_2)

#     # Creating parameter dictionary for SVR
#     params_dict_svr = {
#         "kernel" : kernel,
#         "c_val" : c_val,
#         "gamma" : gamma,
#         "epsilon": epsilon
#     }

#     # Training the Support Vector Machine model
#     svr_1, svr_2 = train_svm_regressor(scaled_X_train, scaled_y_train_1, scaled_y_train_2, params_dict_svr)
#     # model evaluation
#     y_pred1, y_pred2 = eveluate_model(svr_1, scaled_X_test, scaler_y1, svr_2, scaler_y2)
#     r2_score_svr1 = r2_score(y_test_1, y_pred1)*100
#     r2_score_svr2 = r2_score(y_test_2, y_pred2)*100
#     # print the results of evaluation        
#     print("#### SVR Heating Load Models were trained and tested with average r2 score of {:0.2f} %".format(r2_score_svr1))
#     print("#### SVR Cooling Load Models were trained and tested with average r2 score of {:0.2f} %".format(r2_score_svr2))

#     print("Done SVR Model Training")

#     # save the models
#     # dump_scalers(scaler_x, scaler_y1, scaler_y2)

#     # SVM Model
#     svr_path_1 = "./models/svr_1.joblib"
#     svr_path_2 = "./models/svr_2.joblib"
#     dump_models(svr_1, svr_2, svr_path_1, svr_path_2)

#     # SVM
#     # Logging parameters
#     mlflow.log_param("SVR_kernel", params_dict_svr["kernel"])
#     mlflow.log_param("SVR_c_value", params_dict_svr["c_val"])
#     mlflow.log_param("SVR_gamma", params_dict_svr["gamma"])
#     mlflow.log_param("SVR_epsilon", params_dict_svr["epsilon"])
#     # Logging Metric
#     mlflow.log_metric("r2_score for SVR heating load model", r2_score_svr1)
#     mlflow.log_metric("r2_score for SVR cooling load model", r2_score_svr2)
#     # Logging model
#     mlflow.sklearn.log_model(svr_1, "model for SVR heating load")
#     mlflow.sklearn.log_model(svr_2, "model for SVR Cooling load")

#     print("Completed mlflow loggings for SVR")

############################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--kernel", type=str, default='linear')
    parser.add_argument("--C", type=float, default=1)
    parser.add_argument("--gamma", type=str, default='scale')
    parser.add_argument("--epsilon", type=float, default=0.2)

    args = parser.parse_args()

    with mlflow.start_run():
        # main(args.max_depth, args.min_samples_split)  # DT
        main(args.n_estimators, args.max_depth, args.min_samples_split) # RF
        # main(args.kernel, args.C, args.gamma, args.epsilon) # SVM


