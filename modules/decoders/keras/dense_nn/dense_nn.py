import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from base_dockex.BaseKerasModel import BaseKerasModel


class DenseNN(BaseKerasModel):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.units = self.params['units']

    def instantiate_model(self):
        self.model = Sequential()
        self.model.add(
            Dense(
                self.units,
                activation='relu',
                input_dim=self.X_train.shape[1]
            )
        )
        self.model.add(Dense(self.y_train.shape[1]))
        self.model.compile(loss='mse', optimizer='adam')

        self.callbacks.append(EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=5,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        ))


if __name__ == '__main__':
    DenseNN(sys.argv).run()
