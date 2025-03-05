import numpy as np
from yaml import safe_load
import pandas as pd
from pathlib import Path
from logger import create_log_path, CustomLogger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, PowerTransformer
from src.features.outliers_removal import OutlierRemover
import joblib
import sys
import logging


## Logging
# logging set logging path
log_file_path = create_log_path('data_preprocessing')
# Create a custome logger
preprocessing_logger = CustomLogger(logger_name='preprocessing_logger',
                             log_filename=log_file_path)
# set logging level
preprocessing_logger.set_log_level(level = logging.INFO)

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
    # preprocessing_logger.save_logs(msg='f"Processing dataset: {dataframe}')

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


def main():
    # current file path
    current_path = Path(__file__)
    # root directory path
    root_path = current_path.parent.parent.parent
    # input path 
    input_path = root_path / 'data' / 'processed' / 'build_features'
    # read from params files 
    with open('params.yaml') as f:
        params = safe_load(f)
    # percentile value 
    percentiles = list(params['data_preprocessing']['percentiles']) 
    # save transformer path 
    save_transformer_path = root_path / 'models' / 'transformer'
    # make directory 
    save_transformer_path.mkdir(exist_ok=True)
    # save the output file path
    save_data_path = root_path / 'data' / 'processed' / 'final'
    # make directory
    save_data_path.mkdir(exist_ok=True)

    for filename in sys.argv[1:]:
        complete_input_path = input_path / filename
        if filename is 'train.csv':
            # read file
            df = read_dataframe(complete_input_path)
            # split input and output data
            X = df.drop(columns=TARGET)
            y = df[TARGET]
            # remove the outlier
            outlier_transformer = remove_outliers(dataframe=X, percentiles=percentiles,
                                                columns_name=COLUMN_NAMES)
            preprocessing_logger.save_logs(msg=f'Outlier is removed from {filename.split(".")[0]} file')
            # save the transformer 
            save_transformer(path = save_transformer / 'outlier.joblib',
                             object=outlier_transformer)
            # transform the data
            df_without_outliers = transform_data(transformer= outlier_transformer,
                                                 data=X)           
            # train the preporssor of the data
            preprocessor = train_preprocessor(data = df_without_outliers)
            # save the preprocessor
            save_transformer(path= save_transformer_path / 'preprocessor.joblib',
                             object=preprocessor)
            # transform the data
            X_trans = transform_data(transformer=preprocessor,
                                     data=X)
            # fit the target transformer
            output_transformer = transform_output(y)
            # transform the target
            y_trans = transform_data(transformer=output_transformer,
                                     data=y.values.reshape(-1,1))
            # save the transformed output to the df
            X_trans['trip_duration'] = y_trans
            # save the output transformer
            save_transformer(path=save_transformer_path / 'output_transformer.joblib',
                             object=output_transformer)
            
            # save the transformed data
            save_dataframe(dataframe=X_trans,
                           save_path=save_data_path / filename)
            
        elif filename == 'val.csv':
            # read file
            df = read_dataframe(complete_input_path)
            # split input and output data
            X = df.drop(columns=TARGET)
            y = df[TARGET]
            #  load the transfomer
            outlier_transformer = joblib.load(save_transformer_path / "outliers.joblib")
            df_without_outliers = transform_data(transformer=outlier_transformer, data = X)

            # load the preprocessor
            preprocessor = joblib.load(save_transformer_path / 'preprocessor.joblib')
            # transform the data
            X_trans = transform_data(transformer=preprocessor,
                                     data=X)
            # load the transformer
            output_transformer = joblib.load(save_transformer_path / 'output_transformer.joblib')
            y_trans = transform_data(transformer=output_transformer, 
                                     data=y.values.reshape(-1,1))
             # save the transformed output to the df
            X_trans['trip_duration'] = y_trans

            #  save the transformed data
            save_dataframe(dataframe=X_trans,
                        save_path=save_data_path / filename)
            
        elif filename == 'test.csv':
            df = read_dataframe(complete_input_path)
            # load the transformer 
            outlier_transformer = joblib.load( save_transformer_path / "outliers.joblib")
            df_without_outliers = transform_data(transformer=outlier_transformer,
                           data=df)
            
            # load the preprocessor 
            preprocessor = joblib.load(save_transformer_path / 'preprocessor.joblib')
            # transform the data
            X_trans = transform_data(transformer=preprocessor,
                                     data=df)
            
            # save the transformed data
            save_dataframe(dataframe = X_trans, 
                           save_path = save_data_path / filename)
if __name__ == '__main__':
    main()
            




    


