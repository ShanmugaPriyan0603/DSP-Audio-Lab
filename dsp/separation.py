"""Source-separation operations based on short-time Fourier masking."""

from typing import Tuple

import librosa  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    """Return a one-dimensional floating-point audio signal."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    return values


def _validate_band(
    low_frequency: float,
    high_frequency: float,
    sample_rate: int,
) -> None:
    """Validate an inclusive frequency band against the Nyquist limit."""
    nyquist = sample_rate / 2
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if low_frequency < 0 or high_frequency <= low_frequency:
        raise ValueError("frequency band must have 0 <= low < high")
    if high_frequency > nyquist:
        raise ValueError(
            f"high frequency must not exceed the Nyquist frequency ({nyquist:g} Hz)"
        )


def create_band_mask(
    stft: np.ndarray,
    sample_rate: int,
    low_frequency: float,
    high_frequency: float,
) -> np.ndarray:
    """Create a binary STFT mask that keeps one inclusive frequency band."""
    _validate_band(low_frequency, high_frequency, sample_rate)
    if stft.ndim != 2 or stft.shape[0] < 2:
        raise ValueError("stft must be a two-dimensional array with frequency bins")

    n_fft = (stft.shape[0] - 1) * 2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    band = (frequencies >= low_frequency) & (frequencies <= high_frequency)
    return np.broadcast_to(band[:, np.newaxis], stft.shape).copy()


def isolate_frequency_band(
    signal: np.ndarray,
    sample_rate: int,
    low_frequency: float = 20.0,
    high_frequency: float = 250.0,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mask an audio signal's STFT and reconstruct the selected band.

    Returns the original STFT, the binary mask, the masked STFT, and the
    reconstructed time-domain signal.
    """
    values = _validate_signal(signal)
    _validate_band(low_frequency, high_frequency, sample_rate)
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive")

    stft = librosa.stft(
        values,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
    )
    mask = create_band_mask(
        stft, sample_rate, low_frequency, high_frequency
    )
    masked_stft = stft * mask
    isolated = librosa.istft(
        masked_stft,
        hop_length=hop_length,
        window="hann",
        length=values.size,
    )
    return stft, mask, masked_stft, isolated