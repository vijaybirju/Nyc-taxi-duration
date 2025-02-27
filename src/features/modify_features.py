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







