import matplotlib.pyplot as plt
import numpy as np  # type: ignore[reportMissingImports]
import librosa.display  # type: ignore[reportMissingImports]


def plot_signal_and_spectrum(
    time,
    signal,
    frequencies,
    magnitude,
    dominant_frequency
):
    """Display time-domain and frequency-domain representations."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # -------------------------
    # Time domain
    # -------------------------
    axes[0].plot(time, signal)

    axes[0].set_title("Time-Domain Signal")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Show only the first 20 ms so the waveform is visible
    axes[0].set_xlim(0, min(0.02, time[-1]))

    # -------------------------
    # Frequency domain
    # -------------------------
    axes[1].plot(frequencies, magnitude)

    axes[1].set_title(
        f"Frequency Spectrum — Dominant Frequency: "
        f"{dominant_frequency:.2f} Hz"
    )

    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True)

    # Human-audible range
    axes[1].set_xlim(0, 5000)

    plt.tight_layout()
    plt.show()

def plot_audio_analysis(
    time,
    signal,
    frequencies,
    magnitude,
    sample_rate
):
    """Display real audio waveform and frequency spectrum."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # -------------------------
    # Waveform
    # -------------------------

    axes[0].plot(time, signal)

    axes[0].set_title("Audio Waveform")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Don't display an entire 4-minute song at once.
    # Show the first 10 seconds.
    axes[0].set_xlim(0, min(10, time[-1]))

    # -------------------------
    # Frequency spectrum
    # -------------------------

    axes[1].plot(frequencies, magnitude)

    axes[1].set_title("Frequency Spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].grid(True)

    axes[1].set_xlim(0, min(20000, sample_rate / 2))

    plt.tight_layout()
    plt.show()

def plot_spectrogram(stft, sample_rate, hop_length=512):
    """Display the time-frequency representation of an audio signal."""

    magnitude_db = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max
    )

    plt.figure(figsize=(14, 7))

    librosa.display.specshow(
        magnitude_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz"
    )

    plt.colorbar(format="%+2.0f dB")

    plt.title("Audio Spectrogram — STFT")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")

    plt.ylim(0, min(20000, sample_rate / 2))

    plt.tight_layout()
    plt.show()


def _spectrum_limit(sample_rate):
    """Return a readable upper frequency limit for spectrum plots."""
    return min(20000, sample_rate / 2)


def _plot_waveform(axis, time, signal, title):
    axis.plot(time, signal)
    axis.set_title(title)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.set_xlim(0, min(10, time[-1]))
    axis.grid(True)


def _plot_spectrum(axis, frequencies, magnitude, sample_rate, label, color=None):
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
    axis.plot(frequencies, magnitude_db, label=label, color=color)
    axis.set_xlim(0, _spectrum_limit(sample_rate))
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Magnitude (dB)")
    axis.grid(True)
    axis.legend()


def plot_noise_comparison(time, original, noisy, title="Noise Comparison"):
    """Display original and noisy waveforms on shared time axes."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    _plot_waveform(axes[0], time, original, "Original Waveform")
    _plot_waveform(axes[1], time, noisy, title)
    plt.tight_layout()
    plt.show()


def plot_noise_spectrum_comparison(
    frequencies,
    original_magnitude,
    noisy_magnitude,
    sample_rate,
    title="Spectrum Comparison",
    marked_frequency=None,
):
    """Display original and noisy spectra with an optional interference marker."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    _plot_spectrum(
        axes[0], frequencies, original_magnitude, sample_rate, "Original"
    )
    _plot_spectrum(
        axes[1], frequencies, noisy_magnitude, sample_rate, "Noisy", "tab:red"
    )
    axes[0].set_title("Original Spectrum")
    axes[1].set_title(title)
    if marked_frequency is not None and marked_frequency <= _spectrum_limit(sample_rate):
        for axis in axes:
            axis.axvline(marked_frequency, color="black", linestyle="--")
            axis.text(marked_frequency, axis.get_ylim()[1], f" {marked_frequency:g} Hz")
    plt.tight_layout()
    plt.show()


def plot_cancellation_results(
    time,
    signals,
    spectra,
    sample_rate,
    interference_frequency,
):
    """Display original, noisy, and anti-noise-cancelled signals and spectra."""
    labels = ("Original", "Noisy", "Cancelled")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")
    for row, label in enumerate(labels):
        _plot_waveform(axes[row, 0], time, signals[row], f"{label} Waveform")
        _plot_spectrum(
            axes[row, 1], frequencies=spectra[0], magnitude=spectra[row + 1],
            sample_rate=sample_rate, label=label
        )
        axes[row, 1].set_title(f"{label} Spectrum")
        axes[row, 1].axvline(interference_frequency, color="black", linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_filter_result(
    time,
    original,
    noisy,
    filtered,
    frequencies,
    original_magnitude,
    noisy_magnitude,
    filtered_magnitude,
    sample_rate,
    notch_frequency,
):
    """Display waveform and spectrum changes before and after notch filtering."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    _plot_waveform(axes[0], time, original, "Original Waveform")
    axes[0].plot(time[: min(len(time), int(sample_rate * 10))],
                 noisy[: min(len(noisy), int(sample_rate * 10))],
                 alpha=0.65, label="Noisy")
    axes[0].plot(time[: min(len(time), int(sample_rate * 10))],
                 filtered[: min(len(filtered), int(sample_rate * 10))],
                 alpha=0.8, label="Filtered")
    axes[0].legend()

    for magnitude, label, color in (
        (original_magnitude, "Original", "tab:blue"),
        (noisy_magnitude, "Noisy", "tab:red"),
        (filtered_magnitude, "Filtered", "tab:green"),
    ):
        _plot_spectrum(axes[1], frequencies, magnitude, sample_rate, label, color)
    axes[1].set_title(f"Notch Filter Spectrum ({notch_frequency:g} Hz)")
    axes[1].axvline(notch_frequency, color="black", linestyle="--")
    plt.tight_layout()
    plt.show()