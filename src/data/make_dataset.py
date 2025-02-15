import sys
import logging
from yaml import safe_load
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.logger import create_log_path, CustomLogger


log_file_path = create_log_path('make_dataset')
# create custom logger object

dataset_logger = CustomLogger(logger_name='make_dataset',
                              log_filename=log_file_path)

# set the logging level info
dataset_logger.set_log_level(level=logging.INFO)

def load_raw_data(input_path: Path) -> pd.DataFrame:
    raw_data = pd.read_csv(input_path)
    rows, columns = raw_data.shape
    dataset_logger.save_logs(msg=f'{input_path.stem} data read having { rows} rows and {columns} columns'
                             , log_level='info')
    return raw_data






