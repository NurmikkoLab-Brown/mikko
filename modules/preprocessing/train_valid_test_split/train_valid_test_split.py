import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import numpy as np

from base_dockex.BaseDockex import BaseDockex


class TrainValidTestSplit(BaseDockex):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.train_decimal = self.params["train_decimal"]
        self.valid_decimal = self.params["valid_decimal"]
        self.test_decimal = self.params["test_decimal"]

        if self.train_decimal + self.valid_decimal + self.test_decimal > 1.0:
            raise ValueError("train_decimal + valid_decimal + test_decimal must be <= 1.0")

        self.features = None
        self.targets = None
        self.num_samples = None
        
        self.features_train = None
        self.targets_train = None
        self.features_valid = None
        self.targets_valid = None
        self.features_test = None
        self.targets_test = None

    def load_data(self):
        print('Loading data')
        self.features = np.load(self.input_pathnames["features_npy"])
        self.targets = np.load(self.input_pathnames["targets_npy"])
        if self.features.shape[0] != self.features.shape[0]:
            raise Exception("Features and targets must have the same number of samples (axis 0)")

        self.num_samples = self.features.shape[0]

    def split_data(self):
        print('Splitting data')
        train_begin_index = 0
        train_end_index = int(np.floor(self.train_decimal * self.num_samples))
        valid_end_index = train_end_index + int(np.floor(self.valid_decimal * self.num_samples))

        if self.train_decimal + self.valid_decimal + self.test_decimal == 1.0:
            test_end_index = self.num_samples
        else:
            test_end_index = valid_end_index + int(np.floor(self.test_decimal * self.num_samples))

        self.features_train = self.features[train_begin_index:train_end_index]
        self.targets_train = self.targets[train_begin_index:train_end_index]
        self.features_valid = self.features[train_end_index:valid_end_index]
        self.targets_valid = self.targets[train_end_index:valid_end_index]
        self.features_test = self.features[valid_end_index:test_end_index]
        self.targets_test = self.targets[valid_end_index:test_end_index]

    def write_outputs(self):
        print('Writing output arrays')
        np.save(self.output_pathnames["features_train_npy"], self.features_train)
        np.save(self.output_pathnames["targets_train_npy"], self.targets_train)
        np.save(self.output_pathnames["features_valid_npy"], self.features_valid)
        np.save(self.output_pathnames["targets_valid_npy"], self.targets_valid)
        np.save(self.output_pathnames["features_test_npy"], self.features_test)
        np.save(self.output_pathnames["targets_test_npy"], self.targets_test)

    def run(self):
        print('Running')
        
        self.load_data()
        self.split_data()
        self.write_outputs()

        print('Success')


if __name__ == '__main__':
    TrainValidTestSplit(sys.argv).run()
