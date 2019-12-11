# preprocessing/create_window_features

Converts sequential 2D features of size ```samples x num_features``` into windows of features of size
```samples x window_size x num_features```. This 3D form is useful when utilizing recurrent neural networks.
Additionally, standard "flat" arrays are generated that concatenate all windows across features for a given sample to 
create an array of size ```samples x (window_size * num_features)```.

Windows are centered on the current sample and may stretch forwards and/or backwards in time.

Samples that cannot form a complete window (beginning or end of the dataset) are dropped automatically. This module
requires the corresponding targets dataset as well to ensure samples stay index-matched across features and targets.

For an example JSON configuration file, please see test.json.

__Parameters__

* **samples_before** (int): Required. Include this many samples before the current sample in the window.

* **samples_after** (int): Required. Include this many samples after the current sample in the window.

* **use_current_sample** (bool): Required. Whether to include the current sample in the window.

__Input Pathnames__

* **features_npy** (numpy file): Required. 2D features array.

* **targets_npy** (numpy file): Required. Targets array.

Note that the features and targets arrays should be index-matched by row (i.e., ```axis=0```).

__Output Pathnames__

* **window_features_npy** (numpy file): 3D features array of size ```samples x window_size x num_features```.

* **window_targets_npy** (numpy file): Targets array corresponding to window_features_array.

* **flat_window_features_npy** (numpy file): 2D features array of size ```samples x (window_size * num_features)```.

* **flat_window_targets_npy** (numpy file): Targets array corresponding to flat_window_features_npy.
