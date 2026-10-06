import numpy as np  # type: ignore[reportMissingImports]
from dsp.fft import (
    generate_sine_wave,
    calculate_fft,
    find_dominant_frequency
)
from audio.loader import load_audio, normalize_audio
from dsp.fft import calculate_stft
from dsp.noise import (
    add_sinusoidal_noise,
    add_white_noise,
    apply_notch_filter,
    create_sinusoidal_noise,
    notch_frequency_response,
    play_audio,
    stop_audio,
)
from dsp.metrics import calculate_noise_metrics
from dsp.separation import (
    apply_magnitude_mask,
    isolate_frequency_band,
    separate_harmonic_percussive,
)
from visualization.plots import (
    plot_audio_analysis,
    plot_cancellation_results,
    plot_filter_result,
    plot_noise_comparison,
    plot_noise_spectrum_comparison,
    plot_signal_and_spectrum,
    plot_spectrogram,
    plot_frequency_band_isolation,
    plot_separation_waveforms,
    plot_source_spectra,
)


def print_banner():
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║              DSP AUDIO LAB v1.0              ║")
    print("╠══════════════════════════════════════════════╣")
    print("║                                              ║")
    print("║  [1] Generate Test Signal                    ║")
    print("║  [2] Load Audio                              ║")
    print("║  [3] FFT / Spectrum Analysis                 ║")
    print("║  [4] STFT / Spectrogram                      ║")
    print("║  [5] Noise Lab                               ║")
    print("║  [6] Source Separation Lab                   ║")
    print("║                                              ║")
    print("╚══════════════════════════════════════════════╝")
    print()

def load_and_analyze_audio(analyze=True):
    print()
    print("=== AUDIO ANALYSIS ===")
    print()

    file_path = input(
        "Enter path to audio file: "
    ).strip().strip('"')

    print()
    print("Loading audio...")

    try:
        signal, sample_rate = load_audio(file_path)

    except Exception as e:
        print()
        print(f"ERROR: Could not load audio.")
        print(f"Details: {e}")
        return

    signal = normalize_audio(signal)

    duration = len(signal) / sample_rate

    print()
    print("Audio loaded successfully.")
    print()
    print(f"Sampling rate : {sample_rate} Hz")
    print(f"Samples       : {len(signal):,}")
    print(f"Duration      : {duration:.2f} seconds")
    print(f"Channels      : Mono")
    print()

    if not analyze:
        return

    print("Calculating FFT...")

    frequencies, magnitude = calculate_fft(
        signal,
        sample_rate
    )

    dominant = find_dominant_frequency(
        frequencies,
        magnitude
    )

    print(f"Dominant frequency: {dominant:.2f} Hz")

    time = np.arange(len(signal)) / sample_rate

    print()
    print("Opening visualization...")

    plot_audio_analysis(
        time,
        signal,
        frequencies,
        magnitude,
        sample_rate
    )
def generate_test_signal():
    print()
    print("=== TEST SIGNAL GENERATOR ===")
    print()

    frequency = float(input("Frequency (Hz) [440]: ") or 440)
    duration = float(input("Duration (seconds) [2]: ") or 2)
    sample_rate = int(input("Sampling rate (Hz) [44100]: ") or 44100)

    print()
    print("Generating signal...")

    time, signal = generate_sine_wave(
        frequency=frequency,
        duration=duration,
        sample_rate=sample_rate
    )

    print(f"Samples generated : {len(signal)}")
    print(f"Sampling rate     : {sample_rate} Hz")
    print(f"Signal frequency  : {frequency} Hz")

    print()
    print("Calculating FFT...")

    frequencies, magnitude = calculate_fft(
        signal,
        sample_rate
    )

    dominant = find_dominant_frequency(
        frequencies,
        magnitude
    )

    print(f"Detected frequency: {dominant:.2f} Hz")
    print()
    print("Opening visualization...")

    plot_signal_and_spectrum(
        time,
        signal,
        frequencies,
        magnitude,
        dominant
    )

def analyze_spectrogram():
    print()
    print("=== STFT / SPECTROGRAM ANALYSIS ===")
    print()

    file_path = input(
        "Enter path to audio file: "
    ).strip().strip('"')

    print()
    print("Loading audio...")

    try:
        signal, sample_rate = load_audio(file_path)

    except Exception as e:
        print()
        print(f"ERROR: Could not load audio.")
        print(f"Details: {e}")
        return

    signal = normalize_audio(signal)

    print(f"Sampling rate : {sample_rate} Hz")
    print(f"Samples       : {len(signal):,}")
    print()

    n_fft = 2048
    hop_length = 512

    print("Performing STFT...")
    print(f"FFT window    : {n_fft} samples")
    print(f"Hop length    : {hop_length} samples")
    print()

    stft, magnitude = calculate_stft(
        signal,
        sample_rate,
        n_fft,
        hop_length
    )

    print("STFT complete.")
    print("Opening spectrogram...")

    plot_spectrogram(
        stft,
        sample_rate,
        hop_length
    )


def _load_noise_source():
    """Load and normalize the source signal used by Noise Lab."""
    file_path = input("Enter path to audio file: ").strip().strip('"')
    try:
        signal, sample_rate = load_audio(file_path)
        return normalize_audio(signal), sample_rate
    except Exception as error:
        print(f"ERROR: Could not load audio. Details: {error}")
        return None, None


def _spectra_for(signals, sample_rate):
    """Calculate FFTs for a group of signals."""
    frequencies, first_magnitude = calculate_fft(signals[0], sample_rate)
    magnitudes = [first_magnitude]
    for signal in signals[1:]:
        _, magnitude = calculate_fft(signal, sample_rate)
        magnitudes.append(magnitude)
    return frequencies, magnitudes


def _playback_prompt(signals, sample_rate):
    """Offer explicit playback choices after a Noise Lab visualization."""
    while True:
        choice = input("[P] Play  [S] Stop  [Enter] Continue: ").strip().lower()
        if choice == "p":
            selection = input("Play [1] Original  [2] Noisy  [3] Filtered/Cancelled: ")
            selected = signals.get(selection)
            if selected is None:
                print("Invalid playback selection.")
            else:
                try:
                    play_audio(selected, sample_rate)
                except Exception as error:
                    print(f"ERROR: Could not play audio. Details: {error}")
        elif choice == "s":
            stop_audio()
            print("Playback stopped.")
        elif choice == "":
            return
        else:
            print("Invalid playback command.")


def _print_noise_metrics(original, noisy, processed):
    """Print comparable RMS and SNR measurements for a noise experiment."""
    metrics = calculate_noise_metrics(original, noisy, processed)

    def format_db(value):
        return "inf" if np.isinf(value) else f"{value:.2f}"

    print()
    print("Noise-removal measurements")
    print("=" * 66)
    print(f"{'Measurement':<28}{'Value':>18}{'Unit':>12}")
    print("-" * 66)
    print(f"{'RMS level (original)':<28}{metrics['rms_level']:>18.6f}{'amplitude':>12}")
    print(f"{'Noise RMS (noisy-original)':<28}{metrics['noise_rms']:>18.6f}{'amplitude':>12}")
    print(f"{'SNR before processing':<28}{format_db(metrics['snr_before_db']):>18}{'dB':>12}")
    print(f"{'SNR after processing':<28}{format_db(metrics['snr_after_db']):>18}{'dB':>12}")
    print(f"{'SNR improvement':<28}{format_db(metrics['snr_improvement_db']):>18}{'dB':>12}")
    print("=" * 66)


def _show_noise_comparison(original, noisy, sample_rate, title, marked_frequency=None):
    """Display waveform and spectrum comparisons for a noise operation."""
    time = np.arange(len(original)) / sample_rate
    frequencies, magnitudes = _spectra_for((original, noisy), sample_rate)
    plot_noise_comparison(time, original, noisy, title)
    plot_noise_spectrum_comparison(
        frequencies, magnitudes[0], magnitudes[1], sample_rate,
        title, marked_frequency
    )


def noise_lab():
    """Run the terminal-controlled noise generation and cancellation lab."""
    print()
    print("=== NOISE LAB ===")
    original, sample_rate = _load_noise_source()
    if original is None or sample_rate is None:
        return

    while True:
        print()
        print("[1] Add White Noise")
        print("[2] Add Sinusoidal Noise")
        print("[3] Add 50 Hz Power-Line Hum")
        print("[4] Anti-Noise Cancellation")
        print("[5] Noise Removal using Notch Filter")
        print("[0] Back")
        choice = input("NOISE LAB > ").strip()

        try:
            if choice == "1":
                amplitude = float(input("Noise amplitude [0.1]: ") or 0.1)
                noisy = add_white_noise(original, amplitude)
                _show_noise_comparison(original, noisy, sample_rate, "White Noise")
                _playback_prompt({"1": original, "2": noisy}, sample_rate)
            elif choice == "2":
                frequency = float(input("Noise frequency (Hz) [1000]: ") or 1000)
                amplitude = float(input("Noise amplitude [0.1]: ") or 0.1)
                noisy = add_sinusoidal_noise(original, sample_rate, frequency, amplitude)
                _show_noise_comparison(
                    original, noisy, sample_rate, "Sinusoidal Interference", frequency
                )
                _playback_prompt({"1": original, "2": noisy}, sample_rate)
            elif choice == "3":
                hum_choice = input("[1] 50 Hz  [2] 60 Hz: ").strip()
                frequency = 50.0 if hum_choice == "1" else 60.0 if hum_choice == "2" else 0
                if frequency == 0:
                    print("Invalid hum frequency selection.")
                    continue
                amplitude = float(input("Hum amplitude [0.1]: ") or 0.1)
                noisy = add_sinusoidal_noise(original, sample_rate, frequency, amplitude)
                _show_noise_comparison(original, noisy, sample_rate, "Power-Line Hum", frequency)
                _playback_prompt({"1": original, "2": noisy}, sample_rate)
            elif choice == "4":
                frequency = float(input("Interference frequency (Hz) [1000]: ") or 1000)
                amplitude = float(input("Interference amplitude [0.1]: ") or 0.1)
                noise = create_sinusoidal_noise(len(original), sample_rate, frequency, amplitude)
                noisy = original + noise
                anti_noise = -noise
                cancelled = noisy + anti_noise
                frequencies, magnitudes = _spectra_for(
                    (original, noisy, cancelled), sample_rate
                )
                plot_cancellation_results(
                    np.arange(len(original)) / sample_rate,
                    (original, noisy, cancelled),
                    (frequencies, magnitudes[0], magnitudes[1], magnitudes[2]),
                    sample_rate, frequency
                )
                _print_noise_metrics(original, noisy, cancelled)
                print("Cancellation complete: noisy_signal + anti_noise = original_signal.")
                print("This demonstration assumes that the interfering signal is known exactly and perfectly time-aligned. Real Active Noise Cancellation must account for delay, phase, amplitude and the acoustic/environmental path and commonly uses adaptive filtering.")
                _playback_prompt({"1": original, "2": noisy, "3": cancelled}, sample_rate)
            elif choice == "5":
                frequency = float(input("Notch frequency (Hz) [1000]: ") or 1000)
                Q = float(input("Quality factor Q [30]: ") or 30)
                noisy = add_sinusoidal_noise(original, sample_rate, frequency, 0.1)
                filtered = apply_notch_filter(noisy, sample_rate, frequency, Q)
                frequencies, magnitudes = _spectra_for(
                    (original, noisy, filtered), sample_rate
                )
                plot_filter_result(
                    np.arange(len(original)) / sample_rate, original, noisy, filtered,
                    frequencies, magnitudes[0], magnitudes[1], magnitudes[2],
                    sample_rate, frequency
                )
                response_frequencies, response = notch_frequency_response(
                    sample_rate, frequency, Q
                )
                notch_index = np.argmin(np.abs(response_frequencies - frequency))
                print(f"Filter response near {frequency:g} Hz: {20 * np.log10(max(response[notch_index], 1e-12)):.2f} dB")
                _print_noise_metrics(original, noisy, filtered)
                _playback_prompt({"1": original, "2": noisy, "3": filtered}, sample_rate)
            elif choice == "0":
                return
            else:
                print("Invalid selection.")
        except (ValueError, TypeError) as error:
            print(f"Invalid Noise Lab parameter: {error}")
        except Exception as error:
            print(f"Noise Lab error: {error}")


def source_separation_lab():
    """Run source-separation experiments controlled from the terminal."""
    print()
    print("═══ SOURCE SEPARATION LAB ═══")
    print()
    original, sample_rate = _load_noise_source()
    if original is None or sample_rate is None:
        return

    while True:
        print()
        print("[1] Frequency Band Isolation")
        print("[2] Harmonic / Percussive Separation")
        print("[3] Time-Frequency Masking")
        print("[4] Vocal Isolation")
        print("[5] Compare Source Spectra")
        print("[0] Back")
        choice = input("SOURCE SEPARATION LAB > ").strip()

        if choice == "0":
            return
        try:
            n_fft = 2048
            hop_length = 512
            if choice == "1":
                low_frequency = float(input("Low frequency (Hz) [20]: ") or 20)
                high_frequency = float(input("High frequency (Hz) [250]: ") or 250)
                print("Applying STFT frequency mask...")
                original_stft, _, masked_stft, isolated = isolate_frequency_band(
                    original, sample_rate, low_frequency, high_frequency,
                    n_fft, hop_length,
                )
                plot_frequency_band_isolation(
                    original, isolated, original_stft, masked_stft, sample_rate,
                    hop_length, low_frequency, high_frequency,
                )
                _separation_playback_prompt(
                    {"1": original, "2": isolated}, sample_rate,
                    "Original", "Isolated band",
                )
            elif choice == "2":
                harmonic, percussive, _, _, _ = separate_harmonic_percussive(
                    original, sample_rate, n_fft, hop_length
                )
                plot_separation_waveforms(
                    (original, harmonic, percussive),
                    ("Original", "Harmonic", "Percussive"),
                    "Harmonic / Percussive Separation",
                )
                _separation_playback_prompt(
                    {"1": original, "2": harmonic, "3": percussive},
                    sample_rate, "Original", "Harmonic", "Percussive",
                )
            elif choice == "3":
                threshold_db = float(input("Relative threshold (dB) [-24]: ") or -24)
                selected, residual, _, _ = apply_magnitude_mask(
                    original, sample_rate, threshold_db, n_fft, hop_length
                )
                plot_separation_waveforms(
                    (original, selected, residual),
                    ("Original", "Selected", "Residual"),
                    f"Time-Frequency Masking ({threshold_db:g} dB)",
                )
                _separation_playback_prompt(
                    {"1": original, "2": selected, "3": residual},
                    sample_rate, "Original", "Selected", "Residual",
                )
            elif choice == "4":
                low_frequency = float(input("Vocal band low (Hz) [300]: ") or 300)
                high_frequency = float(input("Vocal band high (Hz) [3400]: ") or 3400)
                _, _, _, vocal = isolate_frequency_band(
                    original, sample_rate, low_frequency, high_frequency,
                    n_fft, hop_length,
                )
                plot_separation_waveforms(
                    (original, vocal), ("Original", "Vocal-band estimate"),
                    "Vocal Isolation (Mono Frequency-Band Estimate)",
                )
                _separation_playback_prompt(
                    {"1": original, "2": vocal}, sample_rate,
                    "Original", "Vocal-band estimate",
                )
            elif choice == "5":
                harmonic, percussive, _, _, _ = separate_harmonic_percussive(
                    original, sample_rate, n_fft, hop_length
                )
                frequencies, magnitudes = _spectra_for(
                    (original, harmonic, percussive), sample_rate
                )
                plot_source_spectra(
                    frequencies, magnitudes,
                    ("Original", "Harmonic", "Percussive"),
                    sample_rate, "Source Spectrum Comparison",
                )
            else:
                print("Invalid selection.")
        except (ValueError, TypeError) as error:
            print(f"Invalid Source Separation parameter: {error}")
        except Exception as error:
            print(f"Source Separation Lab error: {error}")


def _separation_playback_prompt(signals, sample_rate, *labels):
    """Offer playback choices for source-separation results."""
    while True:
        playback = input("[P] Play  [S] Stop  [Enter] Continue: ").strip().lower()
        if playback == "p":
            selection = input("Play " + "  ".join(
                f"[{index + 1}] {label}" for index, label in enumerate(labels)
            ) + ": ").strip()
            selected = signals.get(selection)
            if selected is None:
                print("Invalid playback selection.")
            else:
                try:
                    play_audio(selected, sample_rate)
                except Exception as error:
                    print(f"ERROR: Could not play audio. Details: {error}")
        elif playback == "s":
            stop_audio()
            print("Playback stopped.")
        elif playback == "":
            return
        else:
            print("Invalid playback command.")
def main():
    while True:

        print_banner()

        choice = input("DSP LAB > ").strip()

        if choice == "1":
            generate_test_signal()

        elif choice == "2":
            print()
            load_and_analyze_audio(analyze=False)

        elif choice == "3":
            print()
            load_and_analyze_audio()
    
        elif choice == "4":
            print()
            analyze_spectrogram()

        elif choice == "5":
            noise_lab()

        elif choice == "6":
            source_separation_lab()

        elif choice == "0":
            print()
            print("Shutting down DSP Audio Lab...")
            break

        else:
            print()
            print("Invalid selection.")


if __name__ == "__main__":
    main()