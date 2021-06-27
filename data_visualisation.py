import seaborn as sns
import matplotlib.pyplot as plt
from data_utils import load_csv_data
import pandas as pd
import os
import numpy as np

DEFAULT_PLOT_LIST = ['pair', 'hist']


def read_and_visualise_data(data_path: str, list_of_plots: list = None, list_of_features: list = None,
                            label_name: str = None, save_location: str = None):
    """
    reads data and visualises
    :param data_path: path to data
    :param list_of_plots: list of plots
    :param list_of_features: list of features to plot
    :param label_name: name of label to colour plots
    :param save_location: location to save figures

    :return:
    """
    data_frame, data_features = load_csv_data(data_path)
    visualise_data(data_frame, list_of_plots, list_of_features, label_name, save_location)


def visualise_data(data_frame: pd.DataFrame, list_of_plots: list = None, list_of_features: list = None,
                   label_name: str = None, save_location: str = None):
    """
    reads data abd visualises
    :param save_location: location to save figures
    :param label_name: name of label to colour plots
    :param data_frame: pandas data frame
    :param list_of_plots: list of plots
    :param list_of_features: list of features to plot
    :return:
    """
    plt.style.use('fivethirtyeight')
    # use default set of plots if one specified
    if list_of_plots is None:
        list_of_plots = DEFAULT_PLOT_LIST

    if list_of_features is not None:
        # add label to list of features is not in there
        if label_name not in list_of_features:
            number_of_features = len(list_of_features)
            list_of_features.append(label_name)
        else:
            number_of_features = len(list_of_features) - 1
        data_frame = pd.DataFrame(data_frame, columns=list_of_features)
    else:
        list_of_features = data_frame.columns.values
        if label_name in list_of_features:
            number_of_features = len(list_of_features) - 1
        else:
            number_of_features = len(list_of_features)
    # create save location
    if save_location is not None:
        if not os.path.isdir(save_location):
            os.makedirs(save_location)

    # do plots
    for plot_name in list_of_plots:
        if plot_name == 'pair':
            sns.pairplot(data_frame, hue=label_name)
            if save_location is None:
                plt.show()
            else:
                plt.savefig(os.path.join(save_location, 'pair_plot.png'))
        elif plot_name == 'hist':
            plt.figure(2, figsize=(20, 8))
            plot_num = 0
            for feature_name in list_of_features:
                if feature_name != label_name:
                    plot_num += 1
                    plt.subplot(2, int(np.ceil(number_of_features/2)), plot_num)
                    sns.histplot(data_frame, x=feature_name, hue=label_name)
            if save_location is None:
                plt.show()
            else:
                plt.savefig(os.path.join(save_location, 'histograms.png'))
        else:
            print(f'WARNING: {plot_name} plot not supported only {DEFAULT_PLOT_LIST} supported')


if __name__ == '__main__':
    # tests for visualisations
    test_data_path = 'data/heart_failure_clinical_records_dataset.csv'
    test_list_of_features = None
    test_save_location = './test_data_visualisations/'
    read_and_visualise_data(test_data_path, label_name='DEATH_EVENT', list_of_features=test_list_of_features,
                            save_location=test_save_location)
