import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import warnings
import random
import pathlib
import pandas as pd
import numpy as np
import joblib

from base_dockex.BaseDockex import BaseDockex
from AudioHelper import AudioHelper


class LibrosaMelTransformer(BaseDockex):
    def __init__(self, input_args):
        super().__init__(input_args)

        self.sampling_rate = self.params['sampling_rate']
        self.ref_level_db = self.params['ref_level_db']
        self.power = self.params['power']
        self.griffin_lim_iters = self.params['griffin_lim_iters']
        self.fft_size = self.params['fft_size']
        self.num_mels = self.params['num_mels']
        self.preemphasis = self.params['preemphasis']
        self.max_abs_value = self.params['max_abs_value']
        self.min_level_db = self.params['min_level_db']
        self.frame_shift_ms = self.params['frame_shift_ms']
        self.random_seed = self.params['random_seed']
        self.method = self.params['method']
        self.save_transformer = self.params['save_transformer']

        self.input = None
        self.input_is_dataframe = None
        self.audio_helper = None
        self.output = None

    def set_random_seeds(self):
        print('Setting random seeds')
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def load_data(self):
        print('Loading input')
        file_type = pathlib.Path(self.input_pathnames['input']).suffix

        if file_type == '.npy':
            self.input = np.load(self.input_pathnames['input'])
            self.input_is_dataframe = False

        elif file_type == '.h5':
            self.input = pd.read_hdf(self.input_pathnames['input'], 'data')
            self.input_is_dataframe = True
        else:
            raise ValueError("Input must be file type .npy or .h5")

    def instantiate_transformer(self):
        print('Instantiating transformer')
        if self.sampling_rate is None:
            raise ValueError("User must supply a sampling rate if no transformer_joblib input_pathname is provided.")

        else:
            self.audio_helper = AudioHelper(
                sampling_rate=self.sampling_rate,
                ref_level_db=self.ref_level_db,
                power=self.power,
                griffin_lim_iters=self.griffin_lim_iters,
                fft_size=self.fft_size,
                num_mels=self.num_mels,
                preemphasis=self.preemphasis,
                max_abs_value=self.max_abs_value,
                min_level_db=self.min_level_db,
                frame_shift_ms=self.frame_shift_ms,
                random_seed=self.random_seed
            )

    def mel_transform(self):
        print("Generating mel-spectrogram")
        if self.input_is_dataframe:
            mel_spectrogram = self.audio_helper.melspectrogram(
                self.input.values[:, 0].astype(np.float32)
            )

            bin_step_size = int(
                self.audio_helper.sampling_rate
                / (1.0 / (self.audio_helper.frame_shift_ms / 1000.0))
            )
            target_df_index = self.input.iloc[bin_step_size::bin_step_size].index
            self.output = pd.DataFrame(
                mel_spectrogram[0:len(target_df_index)], index=target_df_index
            )
        
        else:
            mel_spectrogram = self.audio_helper.melspectrogram(
                self.input[:, 0].astype(np.float32)
            )
            self.output = mel_spectrogram

    def load_transformer(self):
        print('Loading transformer')
        self.audio_helper = joblib.load(self.input_pathnames['transformer_joblib'])

    def inverse_mel_transform(self):
        print('Performing inverse mel-transform')
        if self.input_is_dataframe:
            self.output = self.audio_helper.inv_melspectrogram(self.input.values)

        else:
            self.output = self.audio_helper.inv_melspectrogram(self.input)

    def write_output_h5(self):
        print('Writing output h5')
        self.output.to_hdf(self.output_pathnames['mel_h5'], 'data')

    def write_output_array(self):
        print('Writing output array')
        np.save(self.output_pathnames['mel_npy'], self.output)

    def write_output_wav(self):
        self.audio_helper.save_wav(self.output, self.output_pathnames["audio_wav"])

    def write_transformer(self):
        if self.input_pathnames['transformer_joblib'] is not None:
            warnings.warn("User requested save transformer when transformer was already loaded from file. Skipping transformer save.")

        else:
            print('Saving transformer')
            with open(self.output_pathnames['transformer_joblib'], 'wb') as transformer_file:
                joblib.dump(self.audio_helper, transformer_file)

    def run(self):
        print('Running')

        self.set_random_seeds()

        self.load_data()

        if self.input_pathnames['transformer_joblib'] is not None:
            self.load_transformer()
        else:
            self.instantiate_transformer()

        if self.method == 'mel_transform':
            self.mel_transform()
            
            if self.input_is_dataframe:
                self.write_output_h5()
            else:
                self.write_output_array()

        elif self.method == 'inverse_mel_transform':
            self.inverse_mel_transform()
            self.write_output_wav()

        else:
            raise ValueError(f"Received unsupported method: {self.method}")

        if self.save_transformer:
            self.write_transformer()

        print('Success')


if __name__ == '__main__':
    LibrosaMelTransformer(sys.argv).run()
