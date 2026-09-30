import librosa
import numpy as np


def load_audio(file_path):
    """
    Load an audio file and convert it to mono.

    Returns:
        signal      : audio samples
        sample_rate : sampling frequency
    """

    signal, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    return signal, sample_rate


def normalize_audio(signal):
    """Normalize audio amplitude to -1 ... +1."""

    peak = np.max(np.abs(signal))

    if peak == 0:
        return signal

    return signal / peak