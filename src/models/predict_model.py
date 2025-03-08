import sys 
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score


TARGET = 'trip_duration'
model_name = 'xgbreg.joblib'

def load_data(path):
    df = pd.read_csv(path)

    return df


def make_x_y(dataframe:pd.DataFrame,target_column:str):
    df_copy = dataframe.copy()

    X = df_copy.drop(columns=target_column)
    y = df_copy[target_column]

    return X, y


def get_predictions(model,X:pd.DataFrame):
    # get the prediction 
    y_pred = model.predict(X)
    return y_pred

def calculate_r2_score(y_actual,y_prediction):
    score = r2_score(y_true=y_actual,y_pred=y_prediction)
    return score


def main():
    # current file path 
    current_path = Path(__file__)
    # root file path
    root_path = current_path.parent.parent.parent
    for ind in range(1,3):
        # read the input file path
        data_path = root_path / "data/processed/final" / sys.argv[ind]
        # load the data
        data = load_data(data_path)
        # make X and y
        X_test, y_test = make_x_y(dataframe=data, target_column=TARGET)
        # model path 
        model_path = root_path / 'models' / 'models' / model_name
        # load the model
        model = joblib.load(model_path)
        # get predictions from model
        y_pred = get_predictions(model=model,X=X_test)
        # calcuate the r2 score
        score = calculate_r2_score(y_actual=y_test,y_prediction=y_pred)
        
        print(f'\nThe R2 score for dataset {sys.argv[ind]} is {score}')
    
if __name__ == "__main__":
    main()