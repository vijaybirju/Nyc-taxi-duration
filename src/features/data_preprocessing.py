import numpy as np
from yaml import safe_load
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PowerTransformer
from src.features.outliers_removal import OutlierRemover
import joblib
import sys


COLUMN_NAMES = ['pickup_latitude',
                'pickup_longitude',
                'dropoff_latitude',
                'dropoff_longitude']

TARGET = 'trip_duration'

def save_transformer(path,object):
    joblib.dump(value=object,
                filename=path)
    

def remove_outliers(dataframe:pd.DataFrame, percentiles:list,columns_name:list) -> pd.DataFrame:
    df = dataframe.copy()

    outlier_tranformer = OutlierRemover(percentile_values=percentiles,col_subset=columns_name)

    outlier_tranformer.fit(dataframe)

    return outlier_tranformer


def train_preprocessor(data:pd.DataFrame):
    ohe_columns = ['vendor_id']
    standard_scale_columns = ['haversine_distance', 'euclidean_distance',
       'manhattan_distance']
    min_max_scale_columns = ['pickup_longitude',
       'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
    
    preprocessor = ColumnTransformer(transformers=[
        ('one-hot',OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'),ohe_columns),
        ('min-max',MinMaxScaler(),min_max_scale_columns),
        ('standard-scale',StandardScaler(),standard_scale_columns)
    ],remainder='passthrough',verbose_feature_names_out=False,n_jobs=1)

    # set the output as df
    preprocessor.set_output(transform='pandas')
    # fit the preprocessor on the training data
    preprocessor.fit(data)

    return preprocessor


def transform_data(transformer, data:pd.DataFrame):
    # transform data
    data_transformed = transformer.transform(data)

    return data_transformed


def transform_output(target:pd.Series):
    # transform target column
    power_transform = PowerTransformer(method='yeo-johnson',standardize=True)
    # fit and transform the target
    target_transform = power_transform.fit(target.values.reshape(-1,1))

    return target_transform


def read_dataframe(path:Path):
    df = pd.read_csv(path)
    return df

def save_dataframe(dataframe:pd.DataFrame,save_path:Path):
    dataframe.to_csv(save_path,index=False)



