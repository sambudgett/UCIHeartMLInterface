import pandas as pd
import os
import numpy as np
from sklearn.utils import shuffle


def load_csv_data(data_path: str = 'data/heart_failure_clinical_records_dataset.csv') -> [pd.DataFrame, list]:
    """
    function to load data from csv
    :param data_path: path to file
    :return: pandas data frame of data
    """
    # check if file exists
    if not os.path.isfile(data_path):
        warning_message = f'data not found at {data_path}'
        print(warning_message)
        return pd.DataFrame({'warning': [warning_message]})
    else:
        data_frame = pd.read_csv(data_path)
        data_features = data_frame.columns.values
        return data_frame, data_features


def get_data_features(data_frame: pd.DataFrame, list_of_features: list) -> np.array:
    """
    gets specifics columns of data
    :param data_frame: pandas data frame containing all features in data
    :param list_of_features: list of feature names to return
    :return: numpy array of requested data
    """
    data_columns = []
    # get each feature name
    for feature_name in list_of_features:
        if feature_name in data_frame:
            data_columns.append(data_frame[feature_name])
        else:
            print(f'{feature_name} does not exist in data. Ignoring request')
    return np.asarray(data_columns)


def read_data_features(list_of_features: list = None,
                       data_path: str = 'data/heart_failure_clinical_records_dataset.csv') -> [np.array, list]:
    """
    returns numpy array for specified data features
    :param list_of_features: list of feature names to return
    :param data_path: path to file
    :return:
    """
    data_frame, data_features = load_csv_data(data_path)
    # if list of features is None then assume we want all data features in np array
    if list_of_features is None:
        list_of_features = data_features
    return get_data_features(data_frame, list_of_features), list_of_features


def random_training_test_split(input_data: np.array, target: np.array,
                               training_split: float = 0.6) -> [np.array, np.array, np.array, np.array]:
    """
    splits data into numpy array
    :param target: data label
    :param input_data: numpy data [example, features]
    :param training_split: portion of data to use for training (must be between 0 and 1
    :return: training data, training targets, test data and test targets numpy arrays
    """
    # shuffle data and targets
    shuffled_data, shuffled_target = shuffle(input_data, target)

    # split data and targets into training and testing sets
    split_index = int(training_split * len(shuffled_data))
    training_data = shuffled_data[:split_index, :]
    test_data = shuffled_data[split_index:, :]
    training_targets = shuffled_target[:split_index]
    test_targets = shuffled_target[split_index:]

    return training_data, training_targets, test_data, test_targets


if __name__ == '__main__':
    # quick test for loading data
    data, features = read_data_features()
    print(data)

