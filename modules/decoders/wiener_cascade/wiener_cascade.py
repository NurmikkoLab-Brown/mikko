import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys

from base_dockex.BaseJoblibModel import BaseJoblibModel
from WienerCascadeRegression import WienerCascadeRegression


class WienerCascade(BaseJoblibModel):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.degree = self.params['degree']

    def instantiate_model(self):
        self.model = WienerCascadeRegression(degree=self.degree)


if __name__ == '__main__':
    WienerCascade(sys.argv).run()
