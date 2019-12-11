# analysis/regression_metrics

This module calculates the mean correlation between target values and predicted values for training, validation,
and test sets. 

Correlations are averaged across columns (i.e., ```axis=1```) using the Fisher z-transformation to impose additivity 
and to approximate a normal sampling distribution. 

For an example JSON configuration file, please see test.json.

__Parameters__

```None```

__Input Pathnames__

* **targets_train_npy** (numpy file): Required. 2D array of training set target values.

* **predict_train_npy** (numpy file): Required. 2D array of training set predicted values.

* **targets_valid_npy** (numpy file): Required. 2D array of validation set target values.

* **predict_valid_npy** (numpy file): Required. 2D array of validation set predicted values.

* **targets_test_npy** (numpy file): Required. 2D array of testing set target values.

* **predict_test_npy** (numpy file): Required. 2D array of testing set predicted values.

__Output Pathnames__

* **csv** (CSV file): Contains the mean correlation for each dataset as well as lists of the correlations for each 
target. 
