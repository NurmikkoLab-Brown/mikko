"""
A basic auditory experiment that decodes intelligible audio from
multiunit spike count features and mel-spectrogram targets.
This experiment builds multiple decoding models and generates
WAV files of the resulting audio reconstructions for the validation set.
"""

import copy
import pandas as pd

from dockex.core.experiment.ExperimentManager import ExperimentManager


FEATURES_PATHNAME = None  # insert path to spike count features numpy file here
TARGETS_PATHNAME = None  # insert path to audio mel-spectrogram targets numpy file here

TRAIN_DECIMAL = 0.80
VALID_DECIMAL = 0.10
TEST_DECIMAL = 0.10

SAMPLING_RATE = 30000.0

BIN_SECONDS = 0.04
NUM_MEL_COMPONENTS = 128
GRIFFIN_LIM_ITERS = 60
FFT_SIZE = 2048
PREEMPHASIS = 0.97
MAX_ABS_VALUE = 5
MIN_LEVEL_DB = -100

SAMPLES_BEFORE_LIST = [8]
SAMPLES_AFTER_LIST = [8]
USE_CURRENT_SAMPLE_LIST = [True]

KALMAN_FILTER_C = 1.0

WIENER_CASCADE_DEGREE = 4

DENSE_NN_UNITS = 1024
SIMPLE_RNN_UNITS = 1024
GRU_RNN_UNITS = 1024
LSTM_RNN_UNITS = 1024

RANDOM_SEED = 1337

# set to True if using GPU support w/ Dockex
ENABLE_GPUS = False


if __name__ == "__main__":

    print("Running")
    manager = ExperimentManager(
        initial_job_num=0,
        experiment_name_prefix='basic_auditory_experiment'
    )

    if ENABLE_GPUS:
        gpu_credits = 1
    else:
        gpu_credits = 0

    run_dict_list = []
    run_dict = dict()
    
    ######################################################################
    # standardize features and targets
    ######################################################################
    features_preprocess = manager.add_job(
        "modules/preprocessing/sklearn_preprocess",
        params={
            "class": "StandardScaler",
            "kwargs": {"copy": True, "with_mean": True, "with_std": True},
            "method": "fit",
            "train_decimal": TRAIN_DECIMAL,
            "save_model": True
        },
        input_pathnames={"X_npy": FEATURES_PATHNAME},
        skip_input_pathnames=True,
        skip_output_pathnames=["output_npy"],
        save_outputs=True
    )
    run_dict['features_preprocess'] = features_preprocess
    run_dict['features_preprocess_name'] = manager.job_list[-1]['name']

    targets_preprocess = manager.add_job(
        "modules/preprocessing/sklearn_preprocess",
        params={
            "class": "StandardScaler",
            "kwargs": {"copy": True, "with_mean": True, "with_std": False},
            "method": "fit",
            "train_decimal": TRAIN_DECIMAL,
            "save_model": True
        },
        input_pathnames={"X_npy": TARGETS_PATHNAME},
        skip_input_pathnames=True,
        skip_output_pathnames=["output_npy"],
        save_outputs=True
    )
    run_dict['targets_preprocess'] = targets_preprocess
    run_dict['targets_preprocess_name'] = manager.job_list[-1]['name']

    features = manager.add_job(
        "modules/preprocessing/sklearn_preprocess",
        params={
            "method": "transform",
        },
        input_pathnames={"X_npy": FEATURES_PATHNAME, "model_joblib": features_preprocess['model_joblib']},
        skip_input_pathnames=['X_npy'],
        skip_output_pathnames=['model_joblib'],
        save_outputs=True
    )
    run_dict['features'] = features
    run_dict['features_name'] = manager.job_list[-1]['name']

    targets = manager.add_job(
        "modules/preprocessing/sklearn_preprocess",
        params={
            "method": "transform",
        },
        input_pathnames={"X_npy": TARGETS_PATHNAME, "model_joblib": targets_preprocess['model_joblib']},
        skip_input_pathnames=['X_npy'],
        skip_output_pathnames=['model_joblib'],
        save_outputs=True
    )
    run_dict['targets'] = targets
    run_dict['targets_name'] = manager.job_list[-1]['name']

    ######################################################################
    # split data into train / valid / test
    ######################################################################
    train_valid_test = manager.add_job(
        "modules/preprocessing/train_valid_test_split",
        params={
            "train_decimal": TRAIN_DECIMAL,
            "valid_decimal": VALID_DECIMAL,
            "test_decimal": TEST_DECIMAL
        },
        input_pathnames={"features_npy": features["output_npy"], "targets_npy": targets["output_npy"]},
        save_outputs=True
    )
    run_dict['train_valid_test'] = train_valid_test
    run_dict['train_valid_test_name'] = manager.job_list[-1]['name']

    ######################################################################
    # invert mel-spectrogram transformation on targets
    # this generates perfect reconstructions for reference
    ######################################################################
    valid_inverse_targets = manager.add_job(
        "modules/preprocessing/sklearn_preprocess",
        params={
            "method": "inverse_transform"
        },
        input_pathnames={
            "X_npy": train_valid_test['targets_valid_npy'],
            "model_joblib": targets_preprocess["model_joblib"],
        },
        skip_output_pathnames=['model_joblib'],
        save_outputs=True
    )
    run_dict['valid_inverse_targets'] = valid_inverse_targets
    run_dict['valid_inverse_targets_name'] = manager.job_list[-1]['name']

    valid_target_wav = manager.add_job(
        "modules/tasks/auditory/librosa_mel_transformer",
        params={
            "method": "inverse_mel_transform",
            "sampling_rate": SAMPLING_RATE,
            "griffin_lim_iters": GRIFFIN_LIM_ITERS,
            "fft_size": FFT_SIZE,
            "num_mels": NUM_MEL_COMPONENTS,
            "preemphasis": PREEMPHASIS,
            "max_abs_value": MAX_ABS_VALUE,
            "min_level_db": MIN_LEVEL_DB,
            "frame_shift_ms": (BIN_SECONDS * 1000.0),
            "random_seed": RANDOM_SEED
        },
        input_pathnames={"input": valid_inverse_targets["output_npy"]},
        skip_output_pathnames=['mel_h5', 'mel_npy', 'transformer_joblib'],
        save_outputs=True
    )
    run_dict['valid_target_wav'] = valid_target_wav
    run_dict['valid_target_wav_name'] = manager.job_list[-1]['name']

    ######################################################################
    # create window arrays
    ######################################################################
    for samples_before in SAMPLES_BEFORE_LIST:
        run_dict['samples_before'] = samples_before
        
        for samples_after in SAMPLES_AFTER_LIST:
            run_dict['samples_after'] = samples_after
            
            for use_current_sample in USE_CURRENT_SAMPLE_LIST:
                run_dict['use_current_sample'] = use_current_sample

                window_features_targets = manager.add_job(
                    "modules/preprocessing/create_window_features",
                    params={
                        "samples_before": samples_before,
                        "samples_after": samples_after,
                        "use_current_sample": use_current_sample,
                    },
                    input_pathnames={
                        "features_npy": features["output_npy"],
                        "targets_npy": targets["output_npy"],
                    },
                    save_outputs=True
                )
                run_dict['window_features_targets'] = window_features_targets
                run_dict['window_features_targets_name'] = manager.job_list[-1]['name']

                ######################################################################
                # train / valid / test split window arrays
                ######################################################################
                window_train_valid_test = manager.add_job(
                    "modules/preprocessing/train_valid_test_split",
                    params={
                        "train_decimal": TRAIN_DECIMAL,
                        "valid_decimal": VALID_DECIMAL,
                        "test_decimal": TEST_DECIMAL
                    },
                    input_pathnames={
                        "features_npy": window_features_targets["window_features_npy"],
                        "targets_npy": window_features_targets["window_targets_npy"],
                    },
                    save_outputs=True
                )
                run_dict['window_train_valid_test'] = window_train_valid_test
                run_dict['window_train_valid_test_name'] = manager.job_list[-1]['name']

                flat_window_train_valid_test = manager.add_job(
                    "modules/preprocessing/train_valid_test_split",
                    params={
                        "train_decimal": TRAIN_DECIMAL,
                        "valid_decimal": VALID_DECIMAL,
                        "test_decimal": TEST_DECIMAL
                    },
                    input_pathnames={
                        "features_npy": window_features_targets["flat_window_features_npy"],
                        "targets_npy": window_features_targets["flat_window_targets_npy"],
                    },
                    save_outputs=True
                )
                run_dict['flat_window_train_valid_test'] = flat_window_train_valid_test
                run_dict['flat_window_train_valid_test_name'] = manager.job_list[-1]['name']

                ######################################################################
                # run Kalman filter
                ######################################################################
                run_dict['decoder_params'] = {'C': KALMAN_FILTER_C}

                decoder_results = manager.add_job(
                    "modules/decoders/kalman_filter",
                    params={
                        "method": "fit_predict",
                        "C": KALMAN_FILTER_C
                    },
                    input_pathnames={
                        "X_train_npy": train_valid_test["features_train_npy"],
                        "y_train_npy": train_valid_test["targets_train_npy"],
                        "X_valid_npy": train_valid_test["features_valid_npy"],
                        "y_valid_npy": train_valid_test["targets_valid_npy"],
                        "X_test_npy": train_valid_test["features_test_npy"],
                        "y_test_npy": train_valid_test["targets_test_npy"],
                    },
                    skip_output_pathnames=['model_joblib'],
                    save_outputs=True
                )
                run_dict['decoder_results'] = decoder_results
                run_dict['decoder_results_name'] = manager.job_list[-1]['name']

                regression_results = manager.add_job(
                    "modules/analysis/regression_metrics",
                    input_pathnames={
                        "targets_train_npy": train_valid_test["targets_train_npy"],
                        "predict_train_npy": decoder_results['predict_train_npy'],
                        "targets_valid_npy": train_valid_test["targets_valid_npy"],
                        "predict_valid_npy": decoder_results['predict_valid_npy'],
                        "targets_test_npy": train_valid_test["targets_test_npy"],
                        "predict_test_npy": decoder_results['predict_test_npy']
                    },
                    save_outputs=True
                )
                run_dict['regression_results'] = regression_results
                run_dict['regression_results_name'] = manager.job_list[-1]['name']

                valid_inverse_predict = manager.add_job(
                    "modules/preprocessing/sklearn_preprocess",
                    params={
                        "method": "inverse_transform"
                    },
                    input_pathnames={
                        "X_npy": decoder_results['predict_valid_npy'],
                        "model_joblib": targets_preprocess["model_joblib"],
                    },
                    save_outputs=True,
                    skip_output_pathnames=['model_joblib']
                )
                run_dict['valid_inverse_predict'] = valid_inverse_predict
                run_dict['valid_inverse_predict_name'] = manager.job_list[-1]['name']

                predict_valid_wav = manager.add_job(
                    "modules/tasks/auditory/librosa_mel_transformer",
                    params={
                        "method": "inverse_mel_transform",
                        "sampling_rate": SAMPLING_RATE,
                        "griffin_lim_iters": GRIFFIN_LIM_ITERS,
                        "fft_size": FFT_SIZE,
                        "num_mels": NUM_MEL_COMPONENTS,
                        "preemphasis": PREEMPHASIS,
                        "max_abs_value": MAX_ABS_VALUE,
                        "min_level_db": MIN_LEVEL_DB,
                        "frame_shift_ms": (BIN_SECONDS * 1000.0),
                        "random_seed": RANDOM_SEED
                    },
                    input_pathnames={"input": valid_inverse_predict["output_npy"]},
                    skip_output_pathnames=['mel_h5', 'mel_npy', 'transformer_joblib'],
                    save_outputs=True
                )
                run_dict['predict_valid_wav'] = predict_valid_wav
                run_dict['predict_valid_wav_name'] = manager.job_list[-1]['name']

                run_dict_list.append(copy.deepcopy(run_dict))

                ######################################################################
                # run Wiener filter
                ######################################################################
                run_dict['decoder_params'] = None

                decoder_results = manager.add_job(
                    "modules/decoders/wiener_filter",
                    params={
                        "method": "fit_predict"
                    },
                    input_pathnames={
                        "X_train_npy": flat_window_train_valid_test["features_train_npy"],
                        "y_train_npy": flat_window_train_valid_test["targets_train_npy"],
                        "X_valid_npy": flat_window_train_valid_test["features_valid_npy"],
                        "y_valid_npy": flat_window_train_valid_test["targets_valid_npy"],
                        "X_test_npy": flat_window_train_valid_test["features_test_npy"],
                        "y_test_npy": flat_window_train_valid_test["targets_test_npy"],
                    },
                    skip_output_pathnames=['model_joblib'],
                    save_outputs=True
                )
                run_dict['decoder_results'] = decoder_results
                run_dict['decoder_results_name'] = manager.job_list[-1]['name']

                regression_results = manager.add_job(
                    "modules/analysis/regression_metrics",
                    input_pathnames={
                        "targets_train_npy": flat_window_train_valid_test["targets_train_npy"],
                        "predict_train_npy": decoder_results['predict_train_npy'],
                        "targets_valid_npy": flat_window_train_valid_test["targets_valid_npy"],
                        "predict_valid_npy": decoder_results['predict_valid_npy'],
                        "targets_test_npy": flat_window_train_valid_test["targets_test_npy"],
                        "predict_test_npy": decoder_results['predict_test_npy']
                    },
                    save_outputs=True
                )
                run_dict['regression_results'] = regression_results
                run_dict['regression_results_name'] = manager.job_list[-1]['name']

                valid_inverse_predict = manager.add_job(
                    "modules/preprocessing/sklearn_preprocess",
                    params={
                        "method": "inverse_transform"
                    },
                    input_pathnames={
                        "X_npy": decoder_results['predict_valid_npy'],
                        "model_joblib": targets_preprocess["model_joblib"],
                    },
                    save_outputs=True,
                    skip_output_pathnames=['model_joblib']
                )
                run_dict['valid_inverse_predict'] = valid_inverse_predict
                run_dict['valid_inverse_predict_name'] = manager.job_list[-1]['name']

                predict_valid_wav = manager.add_job(
                    "modules/tasks/auditory/librosa_mel_transformer",
                    params={
                        "method": "inverse_mel_transform",
                        "sampling_rate": SAMPLING_RATE,
                        "griffin_lim_iters": GRIFFIN_LIM_ITERS,
                        "fft_size": FFT_SIZE,
                        "num_mels": NUM_MEL_COMPONENTS,
                        "preemphasis": PREEMPHASIS,
                        "max_abs_value": MAX_ABS_VALUE,
                        "min_level_db": MIN_LEVEL_DB,
                        "frame_shift_ms": (BIN_SECONDS * 1000.0),
                        "random_seed": RANDOM_SEED
                    },
                    input_pathnames={"input": valid_inverse_predict["output_npy"]},
                    skip_output_pathnames=['mel_h5', 'mel_npy', 'transformer_joblib'],
                    save_outputs=True
                )
                run_dict['predict_valid_wav'] = predict_valid_wav
                run_dict['predict_valid_wav_name'] = manager.job_list[-1]['name']

                run_dict_list.append(copy.deepcopy(run_dict))

                ######################################################################
                # run Wiener cascade
                ######################################################################
                run_dict['decoder_params'] = {'degree': WIENER_CASCADE_DEGREE}

                decoder_results = manager.add_job(
                    "modules/decoders/wiener_cascade",
                    params={
                        "method": "fit_predict",
                        "degree": WIENER_CASCADE_DEGREE
                    },
                    input_pathnames={
                        "X_train_npy": flat_window_train_valid_test["features_train_npy"],
                        "y_train_npy": flat_window_train_valid_test["targets_train_npy"],
                        "X_valid_npy": flat_window_train_valid_test["features_valid_npy"],
                        "y_valid_npy": flat_window_train_valid_test["targets_valid_npy"],
                        "X_test_npy": flat_window_train_valid_test["features_test_npy"],
                        "y_test_npy": flat_window_train_valid_test["targets_test_npy"],
                    },
                    skip_output_pathnames=['model_joblib'],
                    save_outputs=True
                )
                run_dict['decoder_results'] = decoder_results
                run_dict['decoder_results_name'] = manager.job_list[-1]['name']

                regression_results = manager.add_job(
                    "modules/analysis/regression_metrics",
                    input_pathnames={
                        "targets_train_npy": flat_window_train_valid_test["targets_train_npy"],
                        "predict_train_npy": decoder_results['predict_train_npy'],
                        "targets_valid_npy": flat_window_train_valid_test["targets_valid_npy"],
                        "predict_valid_npy": decoder_results['predict_valid_npy'],
                        "targets_test_npy": flat_window_train_valid_test["targets_test_npy"],
                        "predict_test_npy": decoder_results['predict_test_npy']
                    },
                    save_outputs=True
                )
                run_dict['regression_results'] = regression_results
                run_dict['regression_results_name'] = manager.job_list[-1]['name']

                valid_inverse_predict = manager.add_job(
                    "modules/preprocessing/sklearn_preprocess",
                    params={
                        "method": "inverse_transform"
                    },
                    input_pathnames={
                        "X_npy": decoder_results['predict_valid_npy'],
                        "model_joblib": targets_preprocess["model_joblib"],
                    },
                    save_outputs=True,
                    skip_output_pathnames=['model_joblib']
                )
                run_dict['valid_inverse_predict'] = valid_inverse_predict
                run_dict['valid_inverse_predict_name'] = manager.job_list[-1]['name']

                predict_valid_wav = manager.add_job(
                    "modules/tasks/auditory/librosa_mel_transformer",
                    params={
                        "method": "inverse_mel_transform",
                        "sampling_rate": SAMPLING_RATE,
                        "griffin_lim_iters": GRIFFIN_LIM_ITERS,
                        "fft_size": FFT_SIZE,
                        "num_mels": NUM_MEL_COMPONENTS,
                        "preemphasis": PREEMPHASIS,
                        "max_abs_value": MAX_ABS_VALUE,
                        "min_level_db": MIN_LEVEL_DB,
                        "frame_shift_ms": (BIN_SECONDS * 1000.0),
                        "random_seed": RANDOM_SEED
                    },
                    input_pathnames={"input": valid_inverse_predict["output_npy"]},
                    skip_output_pathnames=['mel_h5', 'mel_npy', 'transformer_joblib'],
                    save_outputs=True
                )
                run_dict['predict_valid_wav'] = predict_valid_wav
                run_dict['predict_valid_wav_name'] = manager.job_list[-1]['name']

                run_dict_list.append(copy.deepcopy(run_dict))

                ######################################################################
                # run Dense NN
                ######################################################################
                run_dict['decoder_params'] = {'units': DENSE_NN_UNITS}

                decoder_results = manager.add_job(
                    "modules/decoders/keras/dense_nn",
                    params={
                        "method": "fit_predict",
                        "units": DENSE_NN_UNITS
                    },
                    input_pathnames={
                        "X_train_npy": flat_window_train_valid_test["features_train_npy"],
                        "y_train_npy": flat_window_train_valid_test["targets_train_npy"],
                        "X_valid_npy": flat_window_train_valid_test["features_valid_npy"],
                        "y_valid_npy": flat_window_train_valid_test["targets_valid_npy"],
                        "X_test_npy": flat_window_train_valid_test["features_test_npy"],
                        "y_test_npy": flat_window_train_valid_test["targets_test_npy"],
                    },
                    skip_output_pathnames=['model_keras'],
                    save_outputs=True,
                    gpu_credits=gpu_credits
                )
                run_dict['decoder_results'] = decoder_results
                run_dict['decoder_results_name'] = manager.job_list[-1]['name']

                regression_results = manager.add_job(
                    "modules/analysis/regression_metrics",
                    input_pathnames={
                        "targets_train_npy": flat_window_train_valid_test["targets_train_npy"],
                        "predict_train_npy": decoder_results['predict_train_npy'],
                        "targets_valid_npy": flat_window_train_valid_test["targets_valid_npy"],
                        "predict_valid_npy": decoder_results['predict_valid_npy'],
                        "targets_test_npy": flat_window_train_valid_test["targets_test_npy"],
                        "predict_test_npy": decoder_results['predict_test_npy']
                    },
                    save_outputs=True
                )
                run_dict['regression_results'] = regression_results
                run_dict['regression_results_name'] = manager.job_list[-1]['name']

                valid_inverse_predict = manager.add_job(
                    "modules/preprocessing/sklearn_preprocess",
                    params={
                        "method": "inverse_transform"
                    },
                    input_pathnames={
                        "X_npy": decoder_results['predict_valid_npy'],
                        "model_joblib": targets_preprocess["model_joblib"],
                    },
                    save_outputs=True,
                    skip_output_pathnames=['model_joblib']
                )
                run_dict['valid_inverse_predict'] = valid_inverse_predict
                run_dict['valid_inverse_predict_name'] = manager.job_list[-1]['name']

                predict_valid_wav = manager.add_job(
                    "modules/tasks/auditory/librosa_mel_transformer",
                    params={
                        "method": "inverse_mel_transform",
                        "sampling_rate": SAMPLING_RATE,
                        "griffin_lim_iters": GRIFFIN_LIM_ITERS,
                        "fft_size": FFT_SIZE,
                        "num_mels": NUM_MEL_COMPONENTS,
                        "preemphasis": PREEMPHASIS,
                        "max_abs_value": MAX_ABS_VALUE,
                        "min_level_db": MIN_LEVEL_DB,
                        "frame_shift_ms": (BIN_SECONDS * 1000.0),
                        "random_seed": RANDOM_SEED
                    },
                    input_pathnames={"input": valid_inverse_predict["output_npy"]},
                    skip_output_pathnames=['mel_h5', 'mel_npy', 'transformer_joblib'],
                    save_outputs=True
                )
                run_dict['predict_valid_wav'] = predict_valid_wav
                run_dict['predict_valid_wav_name'] = manager.job_list[-1]['name']

                run_dict_list.append(copy.deepcopy(run_dict))

                ######################################################################
                # run Simple RNN
                ######################################################################
                run_dict['decoder_params'] = {'units': SIMPLE_RNN_UNITS}

                decoder_results = manager.add_job(
                    "modules/decoders/keras/simple_rnn",
                    params={
                        "method": "fit_predict",
                        "units": SIMPLE_RNN_UNITS
                    },
                    input_pathnames={
                        "X_train_npy": window_train_valid_test["features_train_npy"],
                        "y_train_npy": window_train_valid_test["targets_train_npy"],
                        "X_valid_npy": window_train_valid_test["features_valid_npy"],
                        "y_valid_npy": window_train_valid_test["targets_valid_npy"],
                        "X_test_npy": window_train_valid_test["features_test_npy"],
                        "y_test_npy": window_train_valid_test["targets_test_npy"],
                    },
                    skip_output_pathnames=['model_keras'],
                    save_outputs=True,
                    gpu_credits=gpu_credits
                )
                run_dict['decoder_results'] = decoder_results
                run_dict['decoder_results_name'] = manager.job_list[-1]['name']

                regression_results = manager.add_job(
                    "modules/analysis/regression_metrics",
                    input_pathnames={
                        "targets_train_npy": window_train_valid_test["targets_train_npy"],
                        "predict_train_npy": decoder_results['predict_train_npy'],
                        "targets_valid_npy": window_train_valid_test["targets_valid_npy"],
                        "predict_valid_npy": decoder_results['predict_valid_npy'],
                        "targets_test_npy": window_train_valid_test["targets_test_npy"],
                        "predict_test_npy": decoder_results['predict_test_npy']
                    },
                    save_outputs=True
                )
                run_dict['regression_results'] = regression_results
                run_dict['regression_results_name'] = manager.job_list[-1]['name']

                valid_inverse_predict = manager.add_job(
                    "modules/preprocessing/sklearn_preprocess",
                    params={
                        "method": "inverse_transform"
                    },
                    input_pathnames={
                        "X_npy": decoder_results['predict_valid_npy'],
                        "model_joblib": targets_preprocess["model_joblib"],
                    },
                    save_outputs=True,
                    skip_output_pathnames=['model_joblib']
                )
                run_dict['valid_inverse_predict'] = valid_inverse_predict
                run_dict['valid_inverse_predict_name'] = manager.job_list[-1]['name']

                predict_valid_wav = manager.add_job(
                    "modules/tasks/auditory/librosa_mel_transformer",
                    params={
                        "method": "inverse_mel_transform",
                        "sampling_rate": SAMPLING_RATE,
                        "griffin_lim_iters": GRIFFIN_LIM_ITERS,
                        "fft_size": FFT_SIZE,
                        "num_mels": NUM_MEL_COMPONENTS,
                        "preemphasis": PREEMPHASIS,
                        "max_abs_value": MAX_ABS_VALUE,
                        "min_level_db": MIN_LEVEL_DB,
                        "frame_shift_ms": (BIN_SECONDS * 1000.0),
                        "random_seed": RANDOM_SEED
                    },
                    input_pathnames={"input": valid_inverse_predict["output_npy"]},
                    skip_output_pathnames=['mel_h5', 'mel_npy', 'transformer_joblib'],
                    save_outputs=True
                )
                run_dict['predict_valid_wav'] = predict_valid_wav
                run_dict['predict_valid_wav_name'] = manager.job_list[-1]['name']

                run_dict_list.append(copy.deepcopy(run_dict))

                ######################################################################
                # run GRU RNN
                ######################################################################
                run_dict['decoder_params'] = {'units': GRU_RNN_UNITS}

                decoder_results = manager.add_job(
                    "modules/decoders/keras/gru_rnn",
                    params={
                        "method": "fit_predict",
                        "units": GRU_RNN_UNITS
                    },
                    input_pathnames={
                        "X_train_npy": window_train_valid_test["features_train_npy"],
                        "y_train_npy": window_train_valid_test["targets_train_npy"],
                        "X_valid_npy": window_train_valid_test["features_valid_npy"],
                        "y_valid_npy": window_train_valid_test["targets_valid_npy"],
                        "X_test_npy": window_train_valid_test["features_test_npy"],
                        "y_test_npy": window_train_valid_test["targets_test_npy"],
                    },
                    skip_output_pathnames=['model_keras'],
                    save_outputs=True,
                    gpu_credits=gpu_credits
                )
                run_dict['decoder_results'] = decoder_results
                run_dict['decoder_results_name'] = manager.job_list[-1]['name']

                regression_results = manager.add_job(
                    "modules/analysis/regression_metrics",
                    input_pathnames={
                        "targets_train_npy": window_train_valid_test["targets_train_npy"],
                        "predict_train_npy": decoder_results['predict_train_npy'],
                        "targets_valid_npy": window_train_valid_test["targets_valid_npy"],
                        "predict_valid_npy": decoder_results['predict_valid_npy'],
                        "targets_test_npy": window_train_valid_test["targets_test_npy"],
                        "predict_test_npy": decoder_results['predict_test_npy']
                    },
                    save_outputs=True
                )
                run_dict['regression_results'] = regression_results
                run_dict['regression_results_name'] = manager.job_list[-1]['name']

                valid_inverse_predict = manager.add_job(
                    "modules/preprocessing/sklearn_preprocess",
                    params={
                        "method": "inverse_transform"
                    },
                    input_pathnames={
                        "X_npy": decoder_results['predict_valid_npy'],
                        "model_joblib": targets_preprocess["model_joblib"],
                    },
                    save_outputs=True,
                    skip_output_pathnames=['model_joblib']
                )
                run_dict['valid_inverse_predict'] = valid_inverse_predict
                run_dict['valid_inverse_predict_name'] = manager.job_list[-1]['name']

                predict_valid_wav = manager.add_job(
                    "modules/tasks/auditory/librosa_mel_transformer",
                    params={
                        "method": "inverse_mel_transform",
                        "sampling_rate": SAMPLING_RATE,
                        "griffin_lim_iters": GRIFFIN_LIM_ITERS,
                        "fft_size": FFT_SIZE,
                        "num_mels": NUM_MEL_COMPONENTS,
                        "preemphasis": PREEMPHASIS,
                        "max_abs_value": MAX_ABS_VALUE,
                        "min_level_db": MIN_LEVEL_DB,
                        "frame_shift_ms": (BIN_SECONDS * 1000.0),
                        "random_seed": RANDOM_SEED
                    },
                    input_pathnames={"input": valid_inverse_predict["output_npy"]},
                    skip_output_pathnames=['mel_h5', 'mel_npy', 'transformer_joblib'],
                    save_outputs=True
                )
                run_dict['predict_valid_wav'] = predict_valid_wav
                run_dict['predict_valid_wav_name'] = manager.job_list[-1]['name']

                run_dict_list.append(copy.deepcopy(run_dict))

                ######################################################################
                # run LSTM RNN
                ######################################################################
                run_dict['decoder_params'] = {'units': LSTM_RNN_UNITS}

                decoder_results = manager.add_job(
                    "modules/decoders/keras/lstm_rnn",
                    params={
                        "method": "fit_predict",
                        "units": LSTM_RNN_UNITS
                    },
                    input_pathnames={
                        "X_train_npy": window_train_valid_test["features_train_npy"],
                        "y_train_npy": window_train_valid_test["targets_train_npy"],
                        "X_valid_npy": window_train_valid_test["features_valid_npy"],
                        "y_valid_npy": window_train_valid_test["targets_valid_npy"],
                        "X_test_npy": window_train_valid_test["features_test_npy"],
                        "y_test_npy": window_train_valid_test["targets_test_npy"],
                    },
                    skip_output_pathnames=['model_keras'],
                    save_outputs=True,
                    gpu_credits=gpu_credits
                )
                run_dict['decoder_results'] = decoder_results
                run_dict['decoder_results_name'] = manager.job_list[-1]['name']

                regression_results = manager.add_job(
                    "modules/analysis/regression_metrics",
                    input_pathnames={
                        "targets_train_npy": window_train_valid_test["targets_train_npy"],
                        "predict_train_npy": decoder_results['predict_train_npy'],
                        "targets_valid_npy": window_train_valid_test["targets_valid_npy"],
                        "predict_valid_npy": decoder_results['predict_valid_npy'],
                        "targets_test_npy": window_train_valid_test["targets_test_npy"],
                        "predict_test_npy": decoder_results['predict_test_npy']
                    },
                    save_outputs=True
                )
                run_dict['regression_results'] = regression_results
                run_dict['regression_results_name'] = manager.job_list[-1]['name']

                valid_inverse_predict = manager.add_job(
                    "modules/preprocessing/sklearn_preprocess",
                    params={
                        "method": "inverse_transform"
                    },
                    input_pathnames={
                        "X_npy": decoder_results['predict_valid_npy'],
                        "model_joblib": targets_preprocess["model_joblib"],
                    },
                    save_outputs=True,
                    skip_output_pathnames=['model_joblib']
                )
                run_dict['valid_inverse_predict'] = valid_inverse_predict
                run_dict['valid_inverse_predict_name'] = manager.job_list[-1]['name']

                predict_valid_wav = manager.add_job(
                    "modules/tasks/auditory/librosa_mel_transformer",
                    params={
                        "method": "inverse_mel_transform",
                        "sampling_rate": SAMPLING_RATE,
                        "griffin_lim_iters": GRIFFIN_LIM_ITERS,
                        "fft_size": FFT_SIZE,
                        "num_mels": NUM_MEL_COMPONENTS,
                        "preemphasis": PREEMPHASIS,
                        "max_abs_value": MAX_ABS_VALUE,
                        "min_level_db": MIN_LEVEL_DB,
                        "frame_shift_ms": (BIN_SECONDS * 1000.0),
                        "random_seed": RANDOM_SEED
                    },
                    input_pathnames={"input": valid_inverse_predict["output_npy"]},
                    skip_output_pathnames=['mel_h5', 'mel_npy', 'transformer_joblib'],
                    save_outputs=True
                )
                run_dict['predict_valid_wav'] = predict_valid_wav
                run_dict['predict_valid_wav_name'] = manager.job_list[-1]['name']

                run_dict_list.append(copy.deepcopy(run_dict))

    run_csv_filename = f"run_{manager.experiment_name}.csv"
    pd.DataFrame(run_dict_list).to_csv(f"/tmp/dockex/data/{run_csv_filename}")
    manager.send_to_output_saver(run_csv_filename)

    ######################################################################
    # run the experiment
    ######################################################################
    manager.run()

    print("COMPLETE")
