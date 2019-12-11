# tasks/auditory/librosa_mel_transformer

Transforms audio data into its mel-spectrogram representation or transforms a mel-spectrogram into audio.

The Griffin-Lim algorithm is used for reconstructing phase information when transforming a mel-spectrogram to audio.

For an example JSON configuration file, please see test.json.

__Parameters__

* **method** (str): Required. Must be ```mel_transform``` or ```inverse_mel_transform```.

* **sampling_rate** (float): Default = ```30000.0```.

* **ref_level_db** (int): Audio reference level in dB. Default = ```20```.

* **power** (float): Value applied to spectrogram before Griffin-Lim. Default = ```1.5```.

* **griffin_lim_iters** (int): Number of iterations for Griffin-Lim. Default = ```60```.

* **fft_size** (int): Length of fft. Default = ```2048```.

* **num_mels** (int): Number of mel-bands. Default = ```80```.

* **preemphasis** (float): Preemphasis applied before mel-conversion and after Griffin-Lim. Default = ```0.97```.

* **max_abs_value** (int): Max absolute value for normalization. Default = ```4```.

* **min_level_db** (int): Min level in dB for normalization. Default = ```-100```.

* **frame_shift_ms** (float): Shift size in milliseconds. Default = ```20.0```.

* **random_seed** (int): Random seed for numpy and random libraries. Default = ```1337```.

* **save_transformer** (bool): If ```true```, the fitted transformer will be stored to disk. Default = ```false```.

__Input Pathnames__

* **input** (numpy or HDF5 file): Required. Either audio data or mel-spectrogram array. HDF5 file must have data stored at key ```data```.

* **transformer_joblib** (joblib file): A previously saved transformer.

__Output Pathnames__

* **mel_h5** (HDF5 file): Generated mel-spectrogram HDF5 stored at key ```data```. Only generated if method is 
```mel_transform``` and ```input``` was an HDF5 file.

* **mel_npy** (numpy file): Generated mel-spectrogram array. Only generated if method is ```mel_transform``` and 
```input``` was a numpy array.

* **audio_wav** (WAV file): Generated audio file. Only generated if method is ```inverse_mel_transform```.

* **transformer_joblib** (joblib file): If ```save_transformer``` is ```true```, the saved transformer.
