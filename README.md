# DSP Audio Lab

An educational Python project for demonstrating digital signal processing concepts with terminal controls and Matplotlib visualizations.

## Features

- Test sine-wave generation
- FFT and frequency-spectrum analysis
- STFT spectrograms
- White-noise and sinusoidal-noise generation
- 50/60 Hz power-line hum demonstrations
- Known-signal anti-noise cancellation
- IIR notch filtering with SciPy
- Frequency-band isolation with STFT masking and ISTFT reconstruction
- Harmonic/percussive separation using median-filtered STFT masks
- Relative-magnitude time-frequency masking
- Mono vocal-band isolation estimate and source-spectrum comparison
- Optional audio playback through `sounddevice`

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

Choose `3` for FFT analysis, `4` for a spectrogram, `5` for the terminal-controlled Noise Lab, or `6` for the Source Separation Lab. Source Separation Lab provides frequency-band isolation, harmonic/percussive separation, relative-magnitude time-frequency masking, vocal-band isolation, and source-spectrum comparison. Vocal isolation is an approximation for mono files: it keeps a configurable vocal frequency band rather than using stereo center-channel cancellation. Audio playback is optional and requires an available output device. Provide a local audio-file path when prompted. Audio files and generated output are intentionally ignored by Git; add your own test audio locally under `data/`.

## Project Structure

- `main.py`: terminal control center
- `dsp/`: signal generation, FFT, noise, cancellation, filtering, and source separation
- `audio/`: audio loading and normalization
- `visualization/`: Matplotlib plots
- `data/`: local audio inputs
- `output/`: reserved for generated local output; current visualizations are displayed interactively and are not saved automatically
