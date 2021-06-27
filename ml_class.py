from sklearn import svm, tree, ensemble, neural_network, linear_model, neighbors
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
from data_utils import read_data_features, random_training_test_split, load_csv_data
import optuna
import pickle

DEFAULT_METHODS = [svm.NuSVC(), tree.DecisionTreeClassifier(), ensemble.RandomForestClassifier(),
                   ensemble.AdaBoostClassifier(),
                   neural_network.MLPClassifier(max_iter=1000),
                   linear_model.LogisticRegression(), neighbors.KNeighborsClassifier()]
DEFAULT_NAMES = ['non-linear-svm', 'decision-tree', 'random-forest',
                 'ada-boost', 'mlp', 'log_reg', 'k-nn']
plt.style.use('fivethirtyeight')

FULL_FEATURES = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
                 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
                 'smoking', 'time', 'DEATH_EVENT']


class MachineLearningEnsemble:
    def __init__(self, ml_methods: list = None, ml_names: list = None, method_weighting: np.array = None):
        """
        set up machine learning ensemble. Defaults to ['non-linear-svm', 'decision-tree', 'random-forest',
                 'ada-boost', 'mlp', 'log_reg', 'k-nn']
        :param ml_methods: list of machine learning objects that must have 'fit' and 'predict' methods
        :param ml_names: names to assosiate with ml methods
        :param method_weighting: weights of method (defaults to equal weighting if None)
        """
        if ml_methods is None:
            ml_methods = DEFAULT_METHODS
            ml_names = DEFAULT_NAMES
        self.ml_methods = ml_methods
        self.ml_names = ml_names
        if method_weighting is None:
            method_weighting = np.ones(len(ml_methods))
        self.method_weighting = method_weighting
        self.input_data = None
        self.target = None
        self.data_frame = None
        self.full_list_of_features = None
        self.model_trained_flag = False
        self.list_of_features = None

    def initialise_model_from_data_file(self, list_of_features: list = None,
                                        data_path: str = 'data/heart_failure_clinical_records_dataset.csv',
                                        training_split: float = 0.6):
        """
        Trains model on data from a file
        :param list_of_features: list of features
        :param data_path: path to training data
        :param training_split: split for training validation
        :return:
        """
        if list_of_features is None:
            list_of_features = ['time', 'ejection_fraction', 'serum_creatinine']
        self.list_of_features = list_of_features
        self.data_frame, self.full_list_of_features = load_csv_data(data_path=data_path)
        self.input_data = read_data_features(list_of_features, data_path=data_path)[0].T
        self.target = np.squeeze(read_data_features(["DEATH_EVENT"])[0])
        tr_data, tr_targets, eval_data, eval_targets = random_training_test_split(self.input_data, self.target,
                                                                                  training_split=training_split)
        self.training(tr_data, tr_targets)
        area_under_curve, accuracy, f1_score, mcc_score, fpr, tpr, component_accuracies, component_f1_scores, \
            component_mcc_scores = self.evaluate(eval_data, eval_targets, plot_save_loc='results/current_roc_curve.png')
        self.save_model_to_file()
        return area_under_curve, accuracy, f1_score, mcc_score, fpr, tpr, component_accuracies, component_f1_scores, \
            component_mcc_scores

    def load_model_from_file(self, file_name: str = '/.current_model.pkl'):
        """
        load model from file
        :param file_name: path to file
        :return:
        """
        loaded_model = pickle.load(open(file_name, 'rb'))
        self.ml_methods = loaded_model["ml_methods"]
        self.ml_names = loaded_model["ml_names"]
        self.method_weighting = loaded_model["method_weighting"]

    def save_model_to_file(self, file_name: str = '/.current_model.pkl'):
        """
        save model to file
        :param file_name: path to file
        :return:
        """
        save_model = {"ml_methods": self.ml_methods,
                      "ml_names": self.ml_names,
                      "method_weighting": self.method_weighting}
        pickle.dump(save_model, open(file_name, 'wb'))

    def training(self, training_data: np.array, training_labels: np.array, list_of_features: list = None):
        """
        trains models
        :param training_data: data to train on [examples, features]
        :param training_labels: labels to train on
        :param list_of_features: list of feature names (optional)
        :return:
        """
        for method_num, method in enumerate(self.ml_methods):
            method.fit(training_data, training_labels)
        self.model_trained_flag = True
        if list_of_features is not None:
            self.list_of_features = list_of_features

    def predict(self, test_data: np.array) -> [list, np.array]:
        """
        method to predict class of test data
        :param test_data: data to perform prediction on
        :return: predictions (for each classifier) and combined predications
        """
        predictions = []
        combined_predictions = np.zeros(test_data.shape[0])
        for method_num, method in enumerate(self.ml_methods):
            method_prediction = method.predict(test_data)
            predictions.append(method_prediction)
            combined_predictions += (method_prediction * self.method_weighting[method_num])
        combined_predictions = combined_predictions/len(predictions)
        return predictions, combined_predictions

    def evaluate(self, test_data: np.array, test_labels: np.array, plot_save_loc: str = None) \
            -> [float, float, float, np.array, np.array]:
        """
        evaluate trained model on test data
        :param test_data: test data of the same for trained on
        :param test_labels: test labels to compare with
        :param plot_save_loc: save location to plot
        :return: area_under_curve, accuracy, fpr, tpr
        """
        predictions, combined_predictions = self.predict(test_data)

        area_under_curve = metrics.roc_auc_score(test_labels, combined_predictions)
        fpr, tpr, thresholds = metrics.roc_curve(test_labels, combined_predictions)

        final_predictions = combined_predictions > 0.5
        accuracy = metrics.accuracy_score(test_labels, final_predictions)
        f1_score = metrics.f1_score(test_labels, final_predictions)
        mcc_score = metrics.matthews_corrcoef(test_labels, final_predictions)
        component_accuracies, component_f1_scores, component_mcc_scores = [], [], []
        for prediction in predictions:
            component_accuracies.append(metrics.accuracy_score(test_labels, prediction))
            component_f1_scores.append(metrics.f1_score(test_labels, prediction))
            component_mcc_scores.append(metrics.matthews_corrcoef(test_labels, prediction))
        if plot_save_loc is not None:
            plt.figure(figsize=(12, 12))
            plt.plot(fpr, tpr)
            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            plt.title(f'RoC curve, AUC={area_under_curve:.2f}, accuracy={accuracy:.2f}, f1_score={f1_score:.2f}')
            save_dir = os.path.dirname(plot_save_loc)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            plt.savefig(plot_save_loc)
        return area_under_curve, accuracy, f1_score, mcc_score, fpr, tpr, component_accuracies, component_f1_scores, \
               component_mcc_scores

    def average_results(self, input_data: np.array, target: np.array, repeats: int = 100, plot_save_loc: str = None):
        """
        takes a set of data and trains with different splits for 'repeats' number of time. collates results and
        calculates average classification statistics.
        :param plot_save_loc:
        :param input_data:
        :param target:
        :param repeats:
        :return: list_area_under_curve, list_accuracy, list_f1_score, list_mcc_score, list_fpr, list_tpr
        """
        list_area_under_curve, list_accuracy, list_f1_score, list_mcc_score, list_fpr, list_tpr = [], [], [], [], [], []
        for ii in range(repeats):
            tr_data, tr_targets, eval_data, eval_targets = random_training_test_split(input_data, target, 0.6)
            self.training(tr_data, tr_targets)
            area_under_curve, accuracy, f1_score, mcc_score, fpr, tpr, \
                component_accuracies, component_f1_scores, component_mcc_scores = self.evaluate(eval_data, eval_targets)
            list_area_under_curve.append(area_under_curve)
            list_accuracy.append(accuracy)
            list_f1_score.append(f1_score)
            list_mcc_score.append(mcc_score)
            list_fpr.append(fpr)
            list_tpr.append(tpr)
        # plot roc curves
        if plot_save_loc is not None:
            plt.figure(figsize=(12, 12))
            plt.scatter(np.concatenate(list_fpr), np.concatenate(list_tpr), color='g', marker='.')
            plt.xlabel('false positive rate')
            plt.ylabel('true positive rate')
            plt.title(f'RoC curve, '
                      f'AUC={np.mean(list_area_under_curve):.2f}, '
                      f'accuracy={np.mean(list_accuracy):.2f},'
                      f' f1_score={np.mean(list_f1_score):.2f}')
            save_dir = os.path.dirname(plot_save_loc)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            plt.savefig(plot_save_loc)
        return list_area_under_curve, list_accuracy, list_f1_score, list_mcc_score, list_fpr, list_tpr


def optuna_objective(trial):
    """
    optuina objective function for getting weighting of classifiers
    :param trial: optuna trial
    :return:
    """
    list_of_features = ['time', 'ejection_fraction', 'serum_creatinine']
    test_input_data = read_data_features(list_of_features)[0].T
    test_target = np.squeeze(read_data_features(["DEATH_EVENT"])[0])
    weighting = [trial.suggest_float("w1", 0.0, 1.0),
                 trial.suggest_float("w2", 0.0, 1.0),
                 trial.suggest_float("w3", 0.0, 1.0),
                 trial.suggest_float("w4", 0.0, 1.0),
                 trial.suggest_float("w5", 0.0, 1.0),
                 trial.suggest_float("w6", 0.0, 1.0),
                 trial.suggest_float("w7", 0.0, 1.0)]
    ml_ensemble = MachineLearningEnsemble(method_weighting=weighting)
    list_area_under_curve, list_accuracy, list_f1_score, list_mcc_score, list_fpr, list_tpr = \
        ml_ensemble.average_results(test_input_data, test_target)
    return np.mean(list_area_under_curve)


def run_optuna_search():
    """
    runs optuna experiment
    :return: best paramters
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=100)
    # {'w1': 0.9840128776747143, 'w2': 0.4582438818004794, 'w3': 0.2095562039198376, 'w4': 0.763592820959353,
    # 'w5': 0.04232877054050488, 'w6': 0.5692691774222268}
    return study.best_params


def test():
    """
    test of class and methods
    :return:
    """
    list_of_features = ['time', 'ejection_fraction', 'serum_creatinine']
    test_input_data = read_data_features(list_of_features)[0].T
    test_target = np.squeeze(read_data_features(["DEATH_EVENT"])[0])
    tr_data, tr_targets, eval_data, eval_targets = random_training_test_split(test_input_data, test_target, 0.6)
    ml_ensemble = MachineLearningEnsemble()
    ml_ensemble.training(tr_data, tr_targets)
    ml_ensemble.evaluate(eval_data, eval_targets)
    ml_ensemble.average_results(test_input_data, test_target, plot_save_loc='./results/roc_curve.png')


if __name__ == '__main__':
    test()
