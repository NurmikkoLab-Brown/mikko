import sys
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

from base_dockex.BaseKerasModel import BaseKerasModel


class LSTMRNN(BaseKerasModel):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.units = self.params['units']

    def instantiate_model(self):
        self.model = Sequential()
        self.model.add(
            LSTM(
                self.units,
                input_shape=(self.X_train.shape[1], self.X_train.shape[2])
            )
        )
        self.model.add(Dense(self.y_train.shape[1]))
        self.model.compile(loss='mse', optimizer='rmsprop')

        self.callbacks.append(EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            patience=5,
            verbose=1,
            mode="auto",
            restore_best_weights=True,
        ))


if __name__ == '__main__':
    LSTMRNN(sys.argv).run()
