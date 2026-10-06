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
- Optional audio playback through `sounddevice`

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

Choose `5` for the terminal-controlled Noise Lab or `6` for the Source Separation Lab. In Source Separation Lab, choose Frequency Band Isolation, enter a frequency range such as `20` to `250` Hz, and use the playback prompt to hear the isolated band. Provide a local audio-file path when prompted. Audio files and generated output are intentionally ignored by Git; add your own test audio locally under `data/`.

## Project Structure

- `main.py`: terminal control center
- `dsp/`: signal generation, FFT, noise, cancellation, filtering, and source separation
- `audio/`: audio loading and normalization
- `visualization/`: Matplotlib plots
- `data/`: local audio inputs
- `output/`: generated local output
