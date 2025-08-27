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
lags = np.linspace(0, 8, n_bins)      # Lag axis (s)
freqs = np.linspace(0, 40, n_freqs)   # Frequency axis (Hz)

# ---------- Function: compute correlation map ----------
def compute_corr_map(X_data, Y_data):
    fmri_signal = Y_data[:, 0]  # single-column time course
    eeg_features = np.mean(X_data, axis=1)  # average across EEG channels
    corr_map = np.zeros((n_freqs, n_bins))
    for i in range(n_freqs):
        for j in range(n_bins):
            corr_map[i, j] = np.corrcoef(eeg_features[:, i, j], fmri_signal)[0, 1]
    return corr_map

# ---------- Compute correlation maps for each network ----------
corr_DAN = compute_corr_map(X_normalized, Y_DAN)
corr_DMN = compute_corr_map(X_normalized, Y_DMN)
corr_DNa = compute_corr_map(X_normalized, Y_DNa)
corr_DNb = compute_corr_map(X_normalized, Y_DNb)

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
vmin = -np.max(np.abs([corr_DAN, corr_DMN, corr_DNa, corr_DNb]))
vmax = np.max(np.abs([corr_DAN, corr_DMN, corr_DNa, corr_DNb]))

# DAN
sns.heatmap(
    corr_DAN, ax=axes[0,0],
    xticklabels=np.round(lags, 1),
    yticklabels=np.round(freqs, 1),
    cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax, cbar=False
)
axes[0,0].set_title('DAN', fontsize=14)
axes[0,0].set_xlabel('Lag (s)')
axes[0,0].set_ylabel('Frequency (Hz)')

# DMN
sns.heatmap(
    corr_DMN, ax=axes[0,1],
    xticklabels=np.round(lags, 1),
    yticklabels=False,
    cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax,
    cbar_kws={'label': 'Correlation'}
)
axes[0,1].set_title('DMN', fontsize=14)
axes[0,1].set_xlabel('Lag (s)')

# DNa
sns.heatmap(
    corr_DNa, ax=axes[1,0],
    xticklabels=np.round(lags, 1),
    yticklabels=np.round(freqs, 1),
    cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax, cbar=False
)
axes[1,0].set_title('DNa', fontsize=14)
axes[1,0].set_xlabel('Lag (s)')
axes[1,0].set_ylabel('Frequency (Hz)')

# DNb
sns.heatmap(
    corr_DNb, ax=axes[1,1],
    xticklabels=np.round(lags, 1),
    yticklabels=False,
    cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax,
    cbar_kws={'label': 'Correlation'}
)
axes[1,1].set_title('DNb', fontsize=14)
axes[1,1].set_xlabel('Lag (s)')

plt.tight_layout()
plt.show()
