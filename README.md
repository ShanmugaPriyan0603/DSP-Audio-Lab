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
- Optional audio playback through `sounddevice`

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

Choose `5` for the terminal-controlled Noise Lab. Provide a local audio-file path when prompted. Audio files and generated output are intentionally ignored by Git; add your own test audio locally under `data/`.

## Project Structure

- `main.py`: terminal control center
- `dsp/`: signal generation, FFT, noise, cancellation, and filtering
- `audio/`: audio loading and normalization
- `visualization/`: Matplotlib plots
- `data/`: local audio inputs
- `output/`: generated local output
