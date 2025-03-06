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
    

def main():
    # current file path
    current_path = Path(__file__)
    # root path
    root_path = current_path.parent.parent.parent
    # training dat path
    training_data_path = root_path / sys.argv[1]
    # load training data into dataframe
    train_data = load_dataframe(training_data_path)
    # make X and y from training data 
    X_train, y_train = make_x_y(train_data,TARGET)
    # make the model object
    regressor = XGBRegressor()
    # train the model
    regressor = train_model(model=regressor,
                            X_train=X_train,
                            y_train=y_train)
    # save the model after training 
    model_output_path = root_path / 'models' / 'models'
    model_output_path.mkdir(exist_ok=True)
    save_model(model=regressor,save_path=model_output_path / 'xgbreg.joblib')


if __name__ == '__main__':
    main()
    

