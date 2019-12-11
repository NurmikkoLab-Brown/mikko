# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import random
import librosa
import librosa.filters
from scipy import signal
from scipy.io import wavfile


class AudioHelper:
    def __init__(
        self,
        sampling_rate=None,
        ref_level_db=None,
        power=None,
        griffin_lim_iters=None,
        fft_size=None,
        num_mels=None,
        preemphasis=None,
        max_abs_value=None,
        min_level_db=None,
        frame_shift_ms=None,
        random_seed=1337
    ):
        super()

        self.__mel_basis = None
        self._inv_mel_basis = None

        self.sampling_rate = int(sampling_rate)
        self.ref_level_db = ref_level_db
        self.power = power
        self.griffin_lim_iters = griffin_lim_iters
        self.fft_size = fft_size
        self.num_freq = (self.fft_size // 2) + 1
        self.num_mels = num_mels
        self.preemphasis = preemphasis
        self.max_abs_value = max_abs_value
        self.min_level_db = min_level_db
        self.frame_shift_ms = frame_shift_ms
        self.hop_size = self.get_hop_size()

        np.random.seed(random_seed)
        random.seed(random_seed)

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.sampling_rate)[0]

    def save_wav(self, wav, path):
        wavfile.write(
            path,
            self.sampling_rate,
            (wav * 32767 / max(0.01, np.max(np.abs(wav)))).astype(np.int16),
        )

    def spectrogram(self, y):
        D = self._stft(self._preemphasis(y))
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S)

    def inv_spectrogram(self, spectrogram):
        S = self._db_to_amp(
            self._denormalize(spectrogram) + self.ref_level_db
        )  # Convert back to linear
        return self._inv_preemphasis(
            self._griffin_lim(S ** self.power)
        )  # Reconstruct phase

    def melspectrogram(self, y):
        D = self._stft(self._preemphasis(y))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D)))
        return np.transpose(self._normalize(S))

    def inv_melspectrogram(self, melspectrogram):
        S = self._mel_to_linear(
            self._db_to_amp(self._denormalize(np.transpose(melspectrogram)))
        )  # Convert back to linear
        return self._inv_preemphasis(self._griffin_lim(S ** 1.5))  # Reconstruct phase

    # Based on https://github.com/librosa/librosa/issues/434
    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        for i in range(self.griffin_lim_iters):
            if i > 0:
                angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _stft(self, y):
        return librosa.stft(y=y, n_fft=self.fft_size, hop_length=self.hop_size)

    def _istft(self, y):
        return librosa.istft(y, hop_length=self.hop_size)

    # Conversions:
    def _linear_to_mel(self, spectrogram):
        if self.__mel_basis is None:
            self.__mel_basis = self._build_mel_basis()
        return np.dot(self.__mel_basis, spectrogram)

    def _mel_to_linear(self, mel_spectrogram):
        if self._inv_mel_basis is None:
            self._inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        return np.maximum(1e-10, np.dot(self._inv_mel_basis, mel_spectrogram))

    def _build_mel_basis(self):
        n_fft = (self.num_freq - 1) * 2
        return librosa.filters.mel(self.sampling_rate, n_fft, n_mels=self.num_mels)

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _preemphasis(self, x):
        return signal.lfilter([1, -self.preemphasis], [1], x)

    def _inv_preemphasis(self, x):
        return signal.lfilter([1], [1, -self.preemphasis], x)

    def _normalize(self, S):
        return np.clip(
            (2 * self.max_abs_value) * ((S - self.min_level_db) / (-self.min_level_db))
            - self.max_abs_value,
            -self.max_abs_value,
            self.max_abs_value,
        )

    def _denormalize(self, D):
        return (
            (np.clip(D, -self.max_abs_value, self.max_abs_value) + self.max_abs_value)
            * -self.min_level_db
            / (2 * self.max_abs_value)
        ) + self.min_level_db

    def get_hop_size(self):
        hop_size = int(self.frame_shift_ms / 1000 * self.sampling_rate)
        return hop_size
