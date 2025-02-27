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


