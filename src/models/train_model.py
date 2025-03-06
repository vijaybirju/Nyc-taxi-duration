import joblib
import sys
import pandas as pd
from yaml import safe_load
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

TARGET = 'trip_duration'

def load_dataframe(path):
    df = pd.read_csv(path)
    return df


def make_x_y(dataframe:pd.DataFrame,target_column:str):
    df_copy = dataframe.copy()

    X = df_copy.drop(columns=target_column)
    y = df_copy[target_column]

    return X, y


def train_model(model, X_train, y_train):
    # fit the model on the data
    model.fit(X_train,y_train)
    return model


def save_model(model, save_path):
    joblib.dump(value=model,
                filename=save_path)

