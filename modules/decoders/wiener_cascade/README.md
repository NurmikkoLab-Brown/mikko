# decoders/wiener_cascade

Fit a Wiener cascade decoder for regression and make predictions.

This module is based on code from the [Kording Lab Neural_Decoding repo.](https://github.com/KordingLab/Neural_Decoding)

For an example JSON configuration file, please see test.json.

__Parameters__

* **method** (str): Required. Method to call of the model. Must be ```fit```, ```fit_predict```, or ```predict```.

* **degree** (int): Required. Polynomial degree.

* **save_model** (bool): Set to ```true``` to save the fitted model as a joblib file. Only utilized if model is not
loaded from file (i.e., method is not ```predict```). Default = ```false```.

* **random_seed** (int): Random seed for numpy and random libraries. Default = ```1337```.

__Input Pathnames__

* **X_train_npy** (numpy file): Required. 2D array of training set features.

* **y_train_npy** (numpy file): Required. 2D array of training set targets.

* **X_valid_npy** (numpy file): Required. 2D array of validation set features.

* **y_valid_npy** (numpy file): Required. 2D array of validation set targets.

* **X_test_npy** (numpy file): Required. 2D array of testing set features.

* **y_test_npy** (numpy file): Required. 2D array of testing set targets.

* **model_joblib** (joblib file): Pretrained model to load. Only utilized if method is ```predict```.

__Output Pathnames__

* **predict_train_npy** (numpy file): 2D array of training set predictions.

* **predict_valid_npy** (numpy file): 2D array of validation set predictions.

* **predict_test_npy** (numpy file): 2D array of testing set predictions.

* **model_joblib** (joblib file): Saved trained model.
