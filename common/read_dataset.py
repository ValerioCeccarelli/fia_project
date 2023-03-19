import pandas as pd
import numpy as np


def read_dataset_with_pandas(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df.rename(columns=lambda x: x.strip())
    df = df.iloc[:, 2:]
    return df


def read_dataset_with_numpy(file_path: str) -> (np.ndarray, np.ndarray):
    df = read_dataset_with_pandas(file_path)
    data = np.array(df)
    x_data = data[:, :-1]
    y_data = data[:, -1]
    return x_data, y_data


def read_dataset_for_classification(file_path: str) -> (np.ndarray, np.ndarray):
    x_data, y_data = read_dataset_with_numpy(file_path)
    y_data = np.array([elem >= 1400 for elem in y_data])
    return x_data, y_data


def read_dataset_for_regression(file_path: str) -> (np.ndarray, np.ndarray):
    return read_dataset_with_numpy(file_path)