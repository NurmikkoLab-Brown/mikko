import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import json
import numpy as np
from math import exp
import pandas as pd

from base_dockex.BaseDockex import BaseDockex


class RegressionMetrics(BaseDockex):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.targets_train = None
        self.predict_train = None
        self.targets_valid = None
        self.predict_valid = None
        self.targets_test = None
        self.predict_test = None
        
        self.results_df = None

    @staticmethod
    def calculate_rho_list(targets, predict):
        rho_list = []
        for i in range(targets.shape[1]):
            rho = np.corrcoef(targets[:, i].T, predict[:, i].T)[0, 1]
            rho_list.append(rho)

        return rho_list

    def list_fisher_transform(self, val_list):
        return [self.fisher_transform(val) for val in val_list]

    @staticmethod
    def fisher_transform(r):
        # for numerical stability
        if r == 1.0:
            r -= 0.0000000001
        return 0.5 * np.log((1 + r) / (1 - r))

    @staticmethod
    def inverse_fisher_transform(z):
        return (exp(2 * z) - 1) / (exp(2 * z) + 1)

    def mean_rho_from_targets_predict(self, targets, predict):
        rho_list = self.calculate_rho_list(targets, predict)
        fisher_rho_list = self.list_fisher_transform(rho_list)
        mean_fisher_rho = np.mean(fisher_rho_list)
        mean_rho = self.inverse_fisher_transform(mean_fisher_rho)

        return mean_rho, rho_list

    def load_data(self):
        print('Loading data')
        self.targets_train = np.load(self.input_pathnames["targets_train_npy"])
        self.predict_train = np.load(self.input_pathnames["predict_train_npy"])
        self.targets_valid = np.load(self.input_pathnames["targets_valid_npy"])
        self.predict_valid = np.load(self.input_pathnames["predict_valid_npy"])
        self.targets_test = np.load(self.input_pathnames["targets_test_npy"])
        self.predict_test = np.load(self.input_pathnames["predict_test_npy"])

    def calculate_metrics(self):
        train_mean_rho, train_mean_rho_list = self.mean_rho_from_targets_predict(self.targets_train, self.predict_train)
        valid_mean_rho, valid_mean_rho_list = self.mean_rho_from_targets_predict(self.targets_valid, self.predict_valid)
        test_mean_rho, test_mean_rho_list = self.mean_rho_from_targets_predict(self.targets_test, self.predict_test)

        self.results_df = pd.DataFrame([{
            "train_mean_rho": train_mean_rho,
            "train_mean_rho_list": json.dumps(train_mean_rho_list),
            "valid_mean_rho": valid_mean_rho,
            "valid_mean_rho_list": json.dumps(valid_mean_rho_list),
            "test_mean_rho": test_mean_rho,
            "test_mean_rho_list": json.dumps(test_mean_rho_list),
            "name": self.config['name'],
            "regression_metrics_input_pathnames": json.dumps(self.input_pathnames),
            "regression_metrics_output_pathnames": json.dumps(self.output_pathnames)
        }])

    def write_output(self):
        print('Writing output')
        self.results_df.to_csv(self.output_pathnames['csv'])

    def run(self):
        print('Running')
        
        self.load_data()
        self.calculate_metrics()
        self.write_output()

        print('Success')


if __name__ == '__main__':
    RegressionMetrics(sys.argv).run()
