# preprocessing/train_valid_test_split

This module splits index-matched target and feature arrays into training, validation, and test sets. The data is split
sequentially along the rows (i.e., ```axis=0```).

For an example JSON configuration file, please see test.json.

__Parameters__

* **train_decimal** (float): Required. Training data decimal fraction.

* **valid_decimal** (float): Required. Validation data decimal fraction.

* **test_decimal** (float): Required. Testing data decimal fraction.

Note that ```train_decimal + valid_decimal + test_decimal <= 1.0``` 

__Input Pathnames__

* **features_npy** (numpy file): Required. Features array.

* **targets_npy** (numpy file): Required. Targets array.

Note that the features and targets arrays should be index-matched by row (i.e., ```axis=0```).

__Output Pathnames__

* **features_train_npy** (numpy file): Training set features.

* **targets_train_npy** (numpy file): Training set targets.

* **features_valid_npy** (numpy file): Validation set features.

* **targets_valid_npy** (numpy file): Validation set targets.

* **features_test_npy** (numpy file): Testing set features.

* **targets_test_npy** (numpy file): Testing set targets.
