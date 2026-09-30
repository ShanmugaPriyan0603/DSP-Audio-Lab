import numpy as np  # type: ignore[reportMissingImports]


def generate_sine_wave(
    frequency=440,
    duration=2.0,
    sample_rate=44100,
    amplitude=1.0
):
    """Generate a pure sine wave."""
    t = np.arange(0, duration, 1 / sample_rate)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return t, signal


def calculate_fft(signal, sample_rate):
    """Calculate the one-sided FFT magnitude spectrum."""
    n = len(signal)

    # FFT
    fft_result = np.fft.fft(signal)

    # Corresponding frequencies
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)

    # Only keep positive frequencies
    positive = frequencies >= 0

    frequencies = frequencies[positive]
    magnitude = np.abs(fft_result[positive]) / n

    # Correct amplitude for one-sided spectrum
    if len(magnitude) > 2:
        magnitude[1:-1] *= 2

    return frequencies, magnitude


def find_dominant_frequency(frequencies, magnitude):
    """Find the frequency with the greatest magnitude."""
    peak_index = np.argmax(magnitude)
    return frequencies[peak_index]

def calculate_stft(signal, sample_rate, n_fft=2048, hop_length=512):
    """
    Calculate the Short-Time Fourier Transform.

    n_fft:
        Number of samples in each FFT window.

    hop_length:
        Number of samples between successive windows.
    """

    import librosa

    stft = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann"
    )

    magnitude = np.abs(stft)

    return stft, magnitude