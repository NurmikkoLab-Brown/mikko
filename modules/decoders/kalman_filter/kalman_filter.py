import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys

from base_dockex.BaseJoblibModel import BaseJoblibModel
from KalmanFilterRegression import KalmanFilterRegression


class KalmanFilter(BaseJoblibModel):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.C = self.params['C']

    def instantiate_model(self):
        self.model = KalmanFilterRegression(C=self.C)

    def predict(self):
        self.predict_train = self.model.predict(self.X_train, self.y_train)
        self.predict_valid = self.model.predict(self.X_valid, self.y_valid)
        self.predict_test = self.model.predict(self.X_test, self.y_test)


if __name__ == '__main__':
    KalmanFilter(sys.argv).run()
