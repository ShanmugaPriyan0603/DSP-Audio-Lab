"""Quantitative measurements for noise-removal demonstrations."""

import numpy as np  # type: ignore[reportMissingImports]


def _rms(signal):
    """Return the root-mean-square level of a signal."""
    values = np.asarray(signal, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


def _snr_db(signal_rms, noise_rms):
    """Return SNR in dB, including infinity for zero residual noise."""
    if noise_rms == 0:
        return float("inf")
    return float(20 * np.log10(signal_rms / noise_rms))


def calculate_noise_metrics(original, noisy, processed):
    """Measure levels and SNR before and after noise removal."""
    original_rms = _rms(original)
    noise_rms = _rms(np.asarray(noisy) - np.asarray(original))
    residual_rms = _rms(np.asarray(processed) - np.asarray(original))
    snr_before_db = _snr_db(original_rms, noise_rms)
    snr_after_db = _snr_db(original_rms, residual_rms)

    return {
        "rms_level": original_rms,
        "noise_rms": noise_rms,
        "snr_before_db": snr_before_db,
        "snr_after_db": snr_after_db,
        "snr_improvement_db": snr_after_db - snr_before_db,
    }