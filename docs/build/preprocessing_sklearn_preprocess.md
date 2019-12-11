# preprocessing/sklearn_preprocess

This module exposes the ```fit```, ```fit_transform```, and ```transform``` methods for any transformer in the
the scikit-learn preprocessing library. 

For an example JSON configuration file, please see test.json.

__Parameters__

* **method** (str): Required. Must be ```fit```, ```fit_transform```, ```transform```, or ```inverse_transform```.

* **class** (str): Required if ```method``` is ```fit``` or ```fit_transform```. The class of the scikit preprocessing
transformer. Must be located in ```sklearn.preprocessing```.

* **kwargs** (dictionary): The kwargs passed to the transformer.

* **train_decimal** (float or null): If not null, the decimal fraction of input data to use to build the transformer.
Data is taken from the beginning of the array.

* **save_model** (bool): If ```true```, the fitted transformer will be stored to disk.

* **random_seed** (int): Random seed for numpy and random.

__Input Pathnames__

* **X_npy** (numpy file): Required. The input array passed to the ```method```.

* **model_joblib** (joblib file): A previously saved transformer for use with the ```transform``` and
```inverse_transform``` methods.

__Output Pathnames__

* **output_npy** (numpy file): The array returned by the ```method``` if one is returned.

* **model_joblib** (joblib file): If ```save_model``` is ```true```, the saved fitted transformer.
