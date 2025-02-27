import sys
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib as Path
from src.logger import create_log_path, CustomLogger

TARGET_COLUMN = 'trip_duration'
PLOT_PATH = Path("reports/figures/target_distribution.png")

## Logging
# logging set logging path
log_file_path = create_log_path('modify_features')
# Create a custome logger
modify_logger = CustomLogger(logger_name='modify_logger',
                             log_filename=log_file_path)
# set logging level
modify_logger.set_log_level(level = logging.INFO)


## Function applied on target columns
def convert_target_to_minute(dataframe:pd.DataFrame,target_column: str) -> pd.DataFrame:
    # conver target to minute
    dataframe.loc[:,target_column]=dataframe[target_column]/60
    modify_logger.save_logs(msg='Target column is converted from second to minutes')
    return dataframe


def drop_above_two_hunderes_minute(dataframe:pd.DataFrame,target_column: str) -> pd.DataFrame:
    # filter the row with target less 200 minutes 
    filter_series = dataframe[target_column] <= 200
    new_dataframe = dataframe[filter_series,:].copy()
    # max value of target column to checkout the outlier are removed
    max_value = new_dataframe[target_column].max()
    modify_logger.save_logs(msg=f'The max value in target column after transformation is {max_value} and the state of tranformatoon is {max_value <= 200}')
    if max_value <= 200:
        return new_dataframe
    else:
        return ValueError('Outlier target values not removed from the data')  
    

def plot_target(dataframe:pd.DataFrame,target_column: str, save_path: str):
    # plot the density plot of the target after tranformation
    sns.kdeplot(data=dataframe, x=target_column)
    plt.title(f"Distribution of {target_column}")
    # save the plot ar the destination path
    plt.savefig(save_path)
    modify_logger.save_logs(msg='Distribution plot saved at destination')


def drop_columns(dataframe:pd.DataFrame) -> pd.DataFrame:
    modify_logger.save_logs(msg=f'Columns in data before removal are {list(dataframe.columns)}')
    # drop the columns from train and val data
    if 'dropoff_datetime' in dataframe.columns:
        columns_to_drop = ['id','dropoff_datetime','store_and_fwd_flag']
        # dropping the cilumns from dataframe
        dataframe_after_removal = dataframe.drop(columns=columns_to_drop)
        list_of_columns_after_removal = list(dataframe_after_removal.columns)
        modify_logger.save_logs(msg=f'Columns in data after removal are {list_of_columns_after_removal}')
        # Verify if the columns is droped
        modify_logger.save_logs(msg=f'Columns {", ".join(columns_to_drop)} dropped from the data varify { columns_to_drop not in list_of_columns_after_removal}')
        return dataframe_after_removal
    # drop the columns from the data 
    else:
        columns_to_drop = ['id','store_and_fwd_flag']
        # dropping the columns from dataframe
        dataframe_after_removal = dataframe.drop(columns=columns_to_drop)
        list_of_columns_after_removal = list(dataframe_after_removal.columns)
        modify_logger.save_logs(f'Columns in data after removal are {list_of_columns_after_removal}')
        # verifying if columns dropped
        modify_logger.save_logs(msg=f"Columns {', '.join(columns_to_drop)} dropped from data  verify={columns_to_drop not in list_of_columns_after_removal}")
        return dataframe_after_removal


def make_date_feature(dataframe:pd.DataFrame) -> pd.DataFrame:
    new_dataframe = dataframe.copy()
    # No. of rows and columns before tranformation 
    orginal_no_of_rows, original_no_of_columns = new_dataframe.shape

    # convert the column to datetime column
    new_dataframe['pickup_datetime'] = pd.to_datetime(new_dataframe['pickup_datetime'])
    modify_logger.save_logs(msg=f'pickup_datetime column is convert to datetime {new_dataframe['pickup_datetime'].dtype}')

    new_dataframe.loc[:, 'pickup_hour'] = new_dataframe['pickup_datetime'].dt.hour
    new_dataframe.loc[:, 'pickup_day'] = new_dataframe['pickup_datetime'].dt.day
    new_dataframe.loc[:, 'pickup_month'] = new_dataframe['pickup_datetime'].dt.month
    new_dataframe.loc[:, 'pickup_weekday'] = new_dataframe['pickup_datetime'].dt.weekday
    new_dataframe.loc[:,'is_weekend'] = new_dataframe.apply(lambda row: row['pickup_day'] >= 5,axis=1).astype('int')

    # drop the redundant date time column
    new_dataframe = new_dataframe.drop(columns=['pickup_datetime'])
    modify_logger.save_logs(msg=f'pickup_datetime column dropped  verify={"pickup_datetime" not in new_dataframe.columns}')
    
    # number of rows and columns after transformation
    transformed_number_of_rows, transformed_number_of_columns = new_dataframe.shape
    modify_logger.save_logs(msg=f'The number of columns increased by 4 {transformed_number_of_columns == (original_no_of_columns + 5 - 1)}')
    modify_logger.save_logs(msg=f'The number of rows remained the same {orginal_no_of_rows == transformed_number_of_rows}')
    return new_dataframe
 
