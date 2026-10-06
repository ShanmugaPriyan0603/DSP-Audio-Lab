"""Noise generation, cancellation, filtering, and audio playback helpers."""

from typing import Tuple

import numpy as np  # type: ignore[reportMissingImports]
from scipy.signal import filtfilt, freqz, iirnotch, lfilter  # type: ignore[reportMissingImports]
import sounddevice as sd  # type: ignore[reportMissingImports]


def _validate_signal(signal: np.ndarray) -> np.ndarray:
    """Return a one-dimensional floating-point copy of an audio signal."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    return values


def _validate_frequency(frequency: float, sample_rate: int) -> None:
    """Validate a frequency against the sampling theorem."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if frequency <= 0 or frequency >= sample_rate / 2:
        raise ValueError("frequency must be between 0 and the Nyquist frequency")


def create_white_noise(length: int, noise_amplitude: float) -> np.ndarray:
    """Create zero-mean Gaussian white noise for a controlled experiment."""
    if length <= 0:
        raise ValueError("length must be positive")
    if noise_amplitude < 0:
        raise ValueError("noise_amplitude must not be negative")
    return np.random.normal(0.0, noise_amplitude, size=length)


def add_white_noise(signal: np.ndarray, noise_amplitude: float) -> np.ndarray:
    """Add zero-mean Gaussian white noise without changing the input signal."""
    original = _validate_signal(signal)
    noise = create_white_noise(original.size, noise_amplitude)
    return original + noise


def remove_known_noise(
    signal: np.ndarray,
    noise: np.ndarray,
    remaining_noise: float = 0.08,
) -> np.ndarray:
    """Remove known noise while retaining a controlled residual."""
    values = _validate_signal(signal)
    noise_values = _validate_signal(noise)
    if values.shape != noise_values.shape:
        raise ValueError("signal and noise must have the same shape")
    if not 0 <= remaining_noise <= 1:
        raise ValueError("remaining_noise must be between 0 and 1")
    return values - (1 - remaining_noise) * noise_values


def reduce_white_noise(
    signal: np.ndarray,
    sample_rate: int,
    reduction_strength: float = 0.25,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Reduce broadband noise with conservative spectral subtraction.

    The noise floor is estimated from the quietest frames in each frequency
    bin, so this method does not require the original clean signal or noise.
    """
    original = _validate_signal(signal)
    if sample_rate <= 0 or n_fft <= 0 or hop_length <= 0:
        raise ValueError("sample_rate, n_fft, and hop_length must be positive")
    if reduction_strength <= 0:
        raise ValueError("reduction_strength must be positive")

    import librosa

    stft = librosa.stft(
        original, n_fft=n_fft, hop_length=hop_length, window="hann"
    )
    power = np.abs(stft) ** 2
    noise_power = np.percentile(power, 60, axis=1, keepdims=True)
    residual_power = np.maximum(power - reduction_strength * noise_power, 0.0)
    gain = np.sqrt(residual_power / (power + 1e-12))
    denoised_stft = stft * gain
    return librosa.istft(
        denoised_stft,
        hop_length=hop_length,
        window="hann",
        length=original.size,
    )


def create_sinusoidal_noise(
    length: int,
    sample_rate: int,
    frequency: float,
    amplitude: float,
) -> np.ndarray:
    """Create a known sinusoidal noise signal for demonstrations."""
    if length <= 0:
        raise ValueError("length must be positive")
    _validate_frequency(frequency, sample_rate)
    if amplitude < 0:
        raise ValueError("amplitude must not be negative")

    time = np.arange(length) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * time)


def add_sinusoidal_noise(
    signal: np.ndarray,
    sample_rate: int,
    frequency: float,
    amplitude: float,
) -> np.ndarray:
    """Add a sinusoid n(t) = A sin(2 pi f t) to an audio signal."""
    original = _validate_signal(signal)
    _validate_frequency(frequency, sample_rate)
    if amplitude < 0:
        raise ValueError("amplitude must not be negative")

    noise = create_sinusoidal_noise(
        original.size, sample_rate, frequency, amplitude
    )
    return original + noise


def apply_notch_filter(
    signal: np.ndarray,
    sample_rate: int,
    notch_frequency: float = 1000.0,
    Q: float = 30.0,
) -> np.ndarray:
    """Remove a narrow band with a second-order IIR notch filter.

    ``Q`` controls the bandwidth: larger values make the notch narrower.
    Zero-phase filtering is used for audio-length signals so the waveform is
    not shifted in time.
    """
    original = _validate_signal(signal)
    _validate_frequency(notch_frequency, sample_rate)
    if Q <= 0:
        raise ValueError("Q must be positive")

    b, a = iirnotch(notch_frequency, Q, fs=sample_rate)
    if original.size <= 3 * max(len(a), len(b)):
        return lfilter(b, a, original)

    pad_length = min(3 * max(len(a), len(b)), original.size - 1)
    return filtfilt(b, a, original, padlen=pad_length)


def notch_frequency_response(
    sample_rate: int,
    notch_frequency: float = 1000.0,
    Q: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return frequency and magnitude response arrays for the notch filter."""
    _validate_frequency(notch_frequency, sample_rate)
    if Q <= 0:
        raise ValueError("Q must be positive")

    b, a = iirnotch(notch_frequency, Q, fs=sample_rate)
    frequencies, response = freqz(b, a, fs=sample_rate)
    return frequencies, np.abs(response)


def play_audio(signal: np.ndarray, sample_rate: int) -> None:
    """Play a signal through the default sounddevice output."""
    values = _validate_signal(signal)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    sd.play(np.clip(values, -1.0, 1.0), sample_rate)


def stop_audio() -> None:
    """Stop any audio currently playing through sounddevice."""
    sd.stop()
