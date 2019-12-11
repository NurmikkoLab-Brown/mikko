import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys

from base_dockex.BaseJoblibModel import BaseJoblibModel
from WienerFilterRegression import WienerFilterRegression


class WienerFilter(BaseJoblibModel):
    def __init__(self, input_args):
        super().__init__(input_args)

    def instantiate_model(self):
        self.model = WienerFilterRegression()


if __name__ == '__main__':
    WienerFilter(sys.argv).run()
