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


def plot_frequency_band_isolation(
    original,
    isolated,
    original_stft,
    masked_stft,
    sample_rate,
    hop_length,
    low_frequency,
    high_frequency,
):
    """Display the source and the result of a frequency-band mask."""
    time = np.arange(len(original)) / sample_rate
    magnitude_db = [
        librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        for stft in (original_stft, masked_stft)
    ]
    maximum_frequency = min(20000, sample_rate / 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _plot_waveform(axes[0, 0], time, original, "Original Waveform")
    _plot_waveform(axes[0, 1], time, isolated, "Isolated Band Waveform")

    for axis, values, title in zip(
        axes[1], magnitude_db, ("Original Spectrogram", "Masked Spectrogram")
    ):
        librosa.display.specshow(
            values,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="hz",
            ax=axis,
        )
        axis.set_title(title)
        axis.set_ylim(0, maximum_frequency)
        axis.axhline(low_frequency, color="tab:orange", linestyle="--")
        axis.axhline(high_frequency, color="tab:orange", linestyle="--")

    fig.suptitle(
        f"Frequency Band Isolation: {low_frequency:g}-{high_frequency:g} Hz",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def plot_separation_waveforms(signals, labels, title):
    """Display source-separation time-domain components."""
    fig, axes = plt.subplots(len(signals), 1, figsize=(14, 3 * len(signals)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, signal, label in zip(axes, signals, labels):
        time = np.arange(len(signal))
        axis.plot(time, signal)
        axis.set_ylabel(label)
        axis.grid(True)
    axes[-1].set_xlabel("Sample")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def plot_source_spectra(frequencies, magnitudes, labels, sample_rate, title):
    """Display multiple source spectra on a shared frequency axis."""
    fig, axis = plt.subplots(figsize=(14, 7))
    for magnitude, label in zip(magnitudes, labels):
        _plot_spectrum(axis, frequencies, magnitude, sample_rate, label)
    axis.set_title(title)
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
    plot_noise_removal_comparison(
        time,
        signals,
        spectra[0],
        spectra[1:],
        sample_rate,
        "Anti-Noise Cancellation",
        interference_frequency,
        processed_label="Cancelled = Noisy + (-Noise)",
    )


def plot_noise_removal_comparison(
    time,
    signals,
    frequencies,
    magnitudes,
    sample_rate,
    title,
    marked_frequency=None,
    attenuation_db=None,
    processed_label="Processed",
):
    """Display aligned original, noisy, and processed signals and spectra."""
    labels = ("Original", "Noisy", processed_label)
    colors = ("tab:blue", "tab:red", "tab:green")
    fig, axes = plt.subplots(
        3, 2, figsize=(14, 11), sharex="col", sharey="col"
    )

    for row, (label, color) in enumerate(zip(labels, colors)):
        _plot_waveform(axes[row, 0], time, signals[row], f"{label} Waveform")
        _plot_spectrum(
            axes[row, 1], frequencies, magnitudes[row], sample_rate,
            label, color
        )
        axes[row, 1].set_title(f"{label} Spectrum")

    waveform_limit = max(
        float(np.max(np.abs(signal))) for signal in signals
    )
    for axis in axes[:, 0]:
        axis.set_ylim(-waveform_limit * 1.1, waveform_limit * 1.1)

    spectrum_db = [
        20 * np.log10(np.maximum(magnitude, 1e-12))
        for magnitude in magnitudes
    ]
    spectrum_min = min(float(np.min(values)) for values in spectrum_db)
    spectrum_max = max(float(np.max(values)) for values in spectrum_db)
    for axis in axes[:, 1]:
        axis.set_ylim(spectrum_min - 3, spectrum_max + 3)

    if marked_frequency is not None and marked_frequency <= _spectrum_limit(sample_rate):
        for axis in axes[:, 1]:
            axis.axvline(
                marked_frequency, color="black", linestyle="--", linewidth=1
            )
            axis.annotate(
                f"Injected noise: {marked_frequency:g} Hz",
                xy=(marked_frequency, 0.95),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=9,
                backgroundcolor="white",
            )
        if attenuation_db is not None:
            axes[2, 1].annotate(
                f"Attenuation: {attenuation_db:.2f} dB",
                xy=(0.98, 0.08),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.5"},
            )

    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
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
    notch_index = np.argmin(np.abs(frequencies - notch_frequency))
    attenuation_db = 20 * np.log10(
        max(filtered_magnitude[notch_index], 1e-12)
        / max(noisy_magnitude[notch_index], 1e-12)
    )
    plot_noise_removal_comparison(
        time,
        (original, noisy, filtered),
        frequencies,
        (original_magnitude, noisy_magnitude, filtered_magnitude),
        sample_rate,
        f"Notch Filter Comparison ({notch_frequency:g} Hz)",
        notch_frequency,
        attenuation_db,
        processed_label="Filtered",
    )