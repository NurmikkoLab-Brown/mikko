import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import numpy as np
import numba

from base_dockex.BaseDockex import BaseDockex

@numba.njit()
def generate_window_features(samples_before,
                             samples_after,
                             use_current_sample,
                             features,
                             window_features):

    num_samples = features.shape[0]

    for current_index in range(samples_before, (num_samples - samples_after)):
        start_index = current_index - samples_before
        end_index = current_index + samples_after

        if use_current_sample:
            window_features[current_index, :, :] = features[start_index:(end_index + 1)]
        else:
            window_features[current_index, 0:samples_before, :] = features[start_index:current_index]
            window_features[current_index, samples_before:(samples_before + samples_after), :] = features[(current_index + 1):(end_index + 1)]

    return window_features


class CreateWindowFeatures(BaseDockex):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.samples_before = self.params["samples_before"]
        self.samples_after = self.params["samples_after"]
        self.use_current_sample = self.params["use_current_sample"]

        self.features = None
        self.targets = None

        self.window_features = None
        self.window_targets = None
        self.flat_window_features = None
        self.flat_window_targets = None

    def load_data(self):
        print('Loading data')
        self.features = np.load(self.input_pathnames["features_npy"])
        self.targets = np.load(self.input_pathnames["targets_npy"])

        if self.features.shape[0] != self.features.shape[0]:
            raise Exception("Features and targets must have the same number of samples (axis 0)")

    def generate_window_arrays(self):
        print('Generating window arrays')

        if self.use_current_sample:
            window_size = self.samples_before + 1 + self.samples_after
        else:
            window_size = self.samples_before + self.samples_after

        window_features = np.full((self.features.shape[0], window_size, self.features.shape[1]), np.nan)

        window_features = generate_window_features(self.samples_before,
                                                   self.samples_after,
                                                   self.use_current_sample,
                                                   self.features,
                                                   window_features)

        if self.samples_after == 0:
            self.window_features = window_features[self.samples_before:]
            self.window_targets = self.targets[self.samples_before:]
        else:
            self.window_features = window_features[self.samples_before:-self.samples_after]
            self.window_targets = self.targets[self.samples_before:-self.samples_after]

        self.flat_window_features = self.window_features.reshape(self.window_features.shape[0], -1)
        self.flat_window_targets = self.window_targets

    def write_outputs(self):
        print('Writing output arrays')
        np.save(self.output_pathnames["window_features_npy"], self.window_features)
        np.save(self.output_pathnames["window_targets_npy"], self.window_targets)
        np.save(self.output_pathnames["flat_window_features_npy"], self.flat_window_features)
        np.save(self.output_pathnames["flat_window_targets_npy"], self.flat_window_targets)

    def run(self):
        print('Running')
        
        self.load_data()
        self.generate_window_arrays()
        self.write_outputs()

        print('Success')


if __name__ == '__main__':
    CreateWindowFeatures(sys.argv).run()
