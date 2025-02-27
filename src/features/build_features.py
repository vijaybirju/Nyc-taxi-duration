import sys
import pandas as pd 
import numpy as np
from pathlib import Path
from distances import euclidean_distance, manhattan_distance, haversine_distance


new_feature_name =  ['haversine_distance',
                     'euclidean_distance',
                     'manhattan_distance']

build_features_list = [haversine_distance,
                       euclidean_distance,
                       manhattan_distance]

def implement_distances(dataframe:pd.DataFrame,
                        lat1:pd.Series,
                        lon1:pd.Series, 
                        lat2:pd.Series,
                        lon2:pd.Series) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for idx, dist_func in enumerate(build_features_list):
        dataframe[new_feature_name[idx]] = dist_func(lat1, lon1, lat2, lon2)

    return dataframe

def read_dataframe(path):
    df = pd.read_csv(path)
    return df


def save_dataframe(dataframe:pd.DataFrame, save_path):
    dataframe.to_csv(save_path,index=False)

if __name__ == "__main__":
    for ind in range(1,4):
        # read the input file from compand line 
        input_file_path = sys.argv[ind]
        # current file path
        current_path = Path(__file__)
        # root path
        root_path = current_path.parent.parent.parent
        # input data path
        data_path = root_path / input_file_path
        # get the file name 
        filename = data_path.parts[-1]
        # build feature
        df = read_dataframe(data_path)
        df = implement_distances(dataframe=df,
                                 lat1=df['pickup_latitude'],
                                 lon1=df['pickup_longitude'],
                                 lat2=df['dropoff_latitude'],
                                 lon2=df['dropoff_longitude'])
        # save the dataframe 
        output_path = root_path / 'data/processed/build_features'
        output_path.mkdir(parents=True,exist_ok=True)

        save_dataframe(df, output_path / filename)