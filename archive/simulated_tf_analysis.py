

import numpy as np
import mne
import matplotlib.pyplot as plt

# --- 1. Simulate raw EEG data ---
sfreq = 1000  # Hz
times = np.arange(0, 100, 1 / sfreq)  # 100 seconds
n_channels = 5
sin_freq = 10  # Simulate a 10 Hz oscillation

# Create sinusoidal data with some noise
data = 0.1 * np.random.randn(n_channels, len(times))  # noise
data[0] += np.sin(2 * np.pi * sin_freq * times)       # add 10 Hz to channel 0

# --- 2. Create info and Raw object ---
ch_names = [f'EEG {i+1:03d}' for i in range(n_channels)]
ch_types = ['eeg'] * n_channels
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(data, info)

# --- 3. Compute TFR directly on raw ---
freqs = np.linspace(1, 40, 40)
n_cycles = freqs / 2

tfr = raw.compute_tfr(
    method='morlet',
    freqs=freqs,
    output='power',
    n_cycles=n_cycles,
    use_fft=True,
    zero_mean=True,
    decim=10,  # Downsample time axis to speed things up
)

# --- 4. Plot TFR for first channel ---
tfr.plot(picks=0, baseline=(None, 0), mode='logratio',
         title=f"TFR of simulated channel {ch_names[0]}")

