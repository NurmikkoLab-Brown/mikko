import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import numpy as np
import joblib
import warnings
import random

from base_dockex.BaseDockex import BaseDockex


class SklearnPreprocess(BaseDockex):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.transformer_class = self.params['class']
        self.kwargs = self.params['kwargs']
        self.method = self.params['method']
        self.train_decimal = self.params['train_decimal']
        self.save_model = self.params['save_model']
        self.random_seed = self.params['random_seed']

        self.X = None
        self.transformer = None
        self.output = None

    def set_random_seeds(self):
        print('Setting random seeds')
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def load_data(self):
        print('Loading X')
        self.X = np.load(self.input_pathnames['X_npy'])

    def instantiate_transformer(self):
        print('Instantiating transformer')
        if self.transformer_class is not None:
            imported = getattr(
                __import__("sklearn.preprocessing", fromlist=[self.transformer_class]),
                self.transformer_class,
            )

            if self.kwargs is not None:
                self.transformer = imported(**self.kwargs)
            else:
                self.transformer = imported()
        else:
            raise ValueError("Must provide params['class'] when calling fit() or fit_transform().")

    def get_fit_array(self):
        if self.train_decimal is not None:
            train_end_index = int(
                np.floor(self.train_decimal * self.X.shape[0])
            )
            fit_array = self.X[0:train_end_index]
        else:
            fit_array = self.X

        return fit_array

    def fit(self):
        print('Fitting transformer on X')
        self.transformer.fit(self.get_fit_array())

    def transform(self):
        print('Transforming X')
        self.output = self.transformer.transform(self.X)

    def fit_transform(self):
        print('Fit_transforming X')
        self.output = self.transformer.fit_transform(self.get_fit_array())

    def load_model(self):
        if self.input_pathnames['model_joblib'] is not None:
            print('Loading transformer')
            self.transformer = joblib.load(self.input_pathnames['model_joblib'])

        else:
            raise ValueError('model_joblib must point to a saved model for method="transform')

    def inverse_transform(self):
        print('Performing inverse transform')
        self.output = self.transformer.inverse_transform(self.X)

    def write_output_array(self):
        print('Writing output array')
        np.save(self.output_pathnames['output_npy'], self.output)

    def write_model(self):
        if self.method == "transform" or self.method == "inverse_transform":
            warnings.warn("User requested save model when model was already loaded from file. Skipping model save.")

        else:
            print('Saving model')
            with open(self.output_pathnames['model_joblib'], 'wb') as model_file:
                joblib.dump(self.transformer, model_file)

    def run(self):
        print('Running')

        self.set_random_seeds()

        self.load_data()

        if self.method == 'fit':
            self.instantiate_transformer()
            self.fit()

        elif self.method == 'fit_transform':
            self.instantiate_transformer()
            self.fit_transform()

        elif self.method == 'transform':
            self.load_model()
            self.transform()

        elif self.method == 'inverse_transform':
            self.load_model()
            self.inverse_transform()

        else:
            raise ValueError(f"Received unsupported method: {self.method}")

        if self.output is not None:
            self.write_output_array()

        if self.save_model:
            self.write_model()

        print('Success')


if __name__ == '__main__':
    SklearnPreprocess(sys.argv).run()
