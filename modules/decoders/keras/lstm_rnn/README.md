# decoders/keras/lstm_rnn

Fit a single-layer LSTM recurrent neural network decoder for regression and make predictions. 
Implemented using [Keras](https://github.com/keras-team/keras) with a 
[TensorFlow](https://github.com/tensorflow/tensorflow) backend.

Training utilizes a validation set to perform early stopping based on validation loss.

This module is built using a GPU-enabled docker image. 
For more information, see [here](https://github.com/NVIDIA/nvidia-docker).

For an example JSON configuration file, please see test.json.

__Parameters__

* **method** (str): Required. Method to call of the model. Must be ```fit```, ```fit_predict```, or ```predict```.

* **units** (int): Required. Number of LSTM cells.

* **batch_size** (int): Training batch size. Default = ```None```. Note that Keras will default to ```32```.

* **epochs** (int): Maximum number of training epochs. Default = ```2048```.

* **save_model** (bool): Set to ```true``` to save the fitted model as a 
[keras HDF5 file](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model). 
Only utilized if model is not loaded from file (i.e., method is not ```predict```). Default = ```false```.

* **random_seed** (int): Random seed for numpy and random libraries. Default = ```1337```.

__Input Pathnames__

* **X_train_npy** (numpy file): Required. 2D array of training set features.

* **y_train_npy** (numpy file): Required. 2D array of training set targets.

* **X_valid_npy** (numpy file): Required. 2D array of validation set features.

* **y_valid_npy** (numpy file): Required. 2D array of validation set targets.

* **X_test_npy** (numpy file): Required. 2D array of testing set features.

* **y_test_npy** (numpy file): Required. 2D array of testing set targets.

* **model_keras** ([keras HDF5 file](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)): 
Pretrained model to load. Only utilized if method is ```predict```.

__Output Pathnames__

* **predict_train_npy** (numpy file): 2D array of training set predictions.

* **predict_valid_npy** (numpy file): 2D array of validation set predictions.

* **predict_test_npy** (numpy file): 2D array of testing set predictions.

* **model_keras** ([keras HDF5 file](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)): 
Saved trained model.
