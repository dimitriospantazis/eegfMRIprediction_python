import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ---------- Load and Preprocess Data ----------
load_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025', 'processed_data', 'eeg_fmri_data.npz')
data = np.load(load_path, allow_pickle=True)
X = data['X']           # shape: (n_samples, n_channels, n_freqs, n_bins)
Y_DAN = data['Y_DAN']   # shape: (n_samples, 1)
Y_DMN = data['Y_DMN']   # shape: (n_samples, 1)
Y_DNa = data['Y_DNa']   # shape: (n_samples, 1)
Y_DNb = data['Y_DNb']   # shape: (n_samples, 1)
subject_ids = data['subject_ids']

n_samples, n_channels, n_freqs, n_bins = X.shape
epsilon = 1e-10

# ---------- Log-transform ----------
X_log = np.log10(X + epsilon)

# ---------- Z-score normalize per channel & frequency ----------
X_normalized = np.zeros_like(X_log)
for ch in range(n_channels):
    for f in range(n_freqs):
        vals = X_log[:, ch, f, :].reshape(n_samples, -1)
        scaler = StandardScaler()
        vals_scaled = scaler.fit_transform(vals)
        X_normalized[:, ch, f, :] = vals_scaled.reshape(n_samples, n_bins)

# ---------- Parameters ----------
bin_size = 1.0                   # seconds per bin (match your compute_pre_event_tfr_segments)
lags = np.arange(0, n_bins * bin_size, bin_size)  # Lag axis (s)
freqs = np.linspace(0, 40, n_freqs)               # Frequency axis (Hz)
freq_ticks = np.arange(0, 41, 10)                 # Mark every 10 Hz

# ---------- Function: compute correlation map ----------
def compute_corr_map(X_data, Y_data):
    fmri_signal = Y_data[:, 0]  # single-column time course
    eeg_features = np.mean(X_data, axis=1)  # average across EEG channels
    corr_map = np.zeros((n_freqs, n_bins))
    for i in range(n_freqs):
        for j in range(n_bins):
            corr_map[i, j] = np.corrcoef(eeg_features[:, i, j], fmri_signal)[0, 1]
    return corr_map.T  # transpose so: shape (n_bins, n_freqs) → (lags, freqs)

# ---------- Compute correlation maps for each network ----------
corr_DAN = compute_corr_map(X_normalized, Y_DAN)
corr_DMN = compute_corr_map(X_normalized, Y_DMN)
corr_DNa = compute_corr_map(X_normalized, Y_DNa)
corr_DNb = compute_corr_map(X_normalized, Y_DNb)

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
vmin = -np.max(np.abs([corr_DAN, corr_DMN, corr_DNa, corr_DNb]))
vmax = np.max(np.abs([corr_DAN, corr_DMN, corr_DNa, corr_DNb]))

def plot_heatmap(ax, corr, title, add_colorbar=False):
    hm = sns.heatmap(
        corr, ax=ax,
        xticklabels=np.round(freqs, 1),
        yticklabels=np.round(lags, 1),
        cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax,
        cbar=add_colorbar,
        cbar_kws={'label': 'Correlation'} if add_colorbar else None
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Lag (s)')
    # Show ticks every 10 Hz
    tick_positions = [np.argmin(np.abs(freqs - ft)) for ft in freq_ticks]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{ft}" for ft in freq_ticks])

# DAN
plot_heatmap(axes[0,0], corr_DAN, 'DAN')
# DMN
plot_heatmap(axes[0,1], corr_DMN, 'DMN')
# DNa
plot_heatmap(axes[1,0], corr_DNa, 'DNa')
# DNb (with colorbar)
plot_heatmap(axes[1,1], corr_DNb, 'DNb', add_colorbar=True)

plt.tight_layout()
plt.show()
