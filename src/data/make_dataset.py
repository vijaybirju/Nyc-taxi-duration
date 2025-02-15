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


def train_val_split(data: pd.DataFrame,
                    test_size: float,
                    random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(data,
                                            test_size=test_size,
                                            random_state=random_state)
    dataset_logger.save_logs(msg = f"Data is split in train and val with shapes { train_data.shape} and {val_data.shape} respectively",
                             log_level = 'info')
    dataset_logger.save_logs(msg = f"The Parametres values are { test_size} for test_size and { random_state} for random_state",
                             log_leve = 'info')
    
    return train_data, val_data


def save_data(data: pd.DataFrame,
              output_path: Path):
    data.to_csv(output_path, index=False)
    dataset_logger.save_logs(msg=f"{output_path.stem + output_path.suffix} data saved successfully to the output folder",
                             log_level='info')
    

def read_params(input_file):
    try:
        with open(input_file) as f:
            params_file = safe_load(f)
    except FileNotFoundError as e:
        dataset_logger.save_logs(msg=f'Parameters value not found switching to defualt values for train test split',
                                 log_level='info')
        defualt_dict = {'test_size': 0.25,
                        'random_state': None}
        # read the defualt dict
        test_size = defualt_dict['test_size']
        random_state = defualt_dict['random_state']

        return test_size, random_state
    else:
        dataset_logger.save_logs(msg=f'Parameters file read succussfully',
                                 log_level='info')
        # read the parameters from the parameters file
        test_size = params_file['make_dataset']['test_size']
        random_state = params_file['make_dataset']['random_state']
        return test_size, random_state
    




