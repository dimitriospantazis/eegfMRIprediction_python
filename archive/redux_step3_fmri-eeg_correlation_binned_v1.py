import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ---------- Load Binned/Band-Averaged Data ----------
load_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025', 'processed_data',
                         'eeg_fmri_data_binned_2s_0to10s_canonicalbands.npz')
data = np.load(load_path, allow_pickle=True)

# UPDATED names to match the new saved file
X_band = data['X_binned']                          # (n_samples, n_channels, n_bands, n_bins) [log-space]
band_names = list(data['bands'])                   # ['delta','theta','alpha','beta','gamma']
time_bin_labels = list(data['time_bin_labels'])    # ['0–2s','2–4s','4–6s','6–8s','8–10s']

Y_DAN = data['Y_DAN']               # (n_samples, 1)
Y_DMN = data['Y_DMN']
Y_DNa = data['Y_DNa']
Y_DNb = data['Y_DNb']
subject_ids = data['subject_ids']

n_samples, n_channels, n_bands, n_bins = X_band.shape

# ---------- Z-score normalize per channel & band ----------
X_normalized = np.zeros_like(X_band)
for ch in range(n_channels):
    for b in range(n_bands):
        vals = X_band[:, ch, b, :].reshape(n_samples, -1)  # (n_samples, n_bins)
        scaler = StandardScaler()
        vals_scaled = scaler.fit_transform(vals)
        X_normalized[:, ch, b, :] = vals_scaled.reshape(n_samples, n_bins)

# ---------- Axes ----------
lags = time_bin_labels               # human-readable 2s bin labels
bands = band_names                   # categorical X axis

# ---------- Function: compute correlation map ----------
def compute_corr_map(X_data, Y_data):
    """
    X_data: (n_samples, n_channels, n_bands, n_bins)
    Y_data: (n_samples, 1)
    Returns: (n_bins, n_bands)  # (lags x bands)
    """
    fmri_signal = Y_data[:, 0]                  # (n_samples,)
    eeg_features = np.mean(X_data, axis=1)      # avg across channels -> (n_samples, n_bands, n_bins)
    corr_map = np.zeros((n_bands, n_bins), dtype=float)
    for b in range(n_bands):
        for t in range(n_bins):
            corr_map[b, t] = np.corrcoef(eeg_features[:, b, t], fmri_signal)[0, 1]
    return corr_map.T                            # (n_bins, n_bands)

# ---------- Compute correlation maps ----------
corr_DAN = compute_corr_map(X_normalized, Y_DAN)
corr_DMN = compute_corr_map(X_normalized, Y_DMN)
corr_DNa = compute_corr_map(X_normalized, Y_DNa)
corr_DNb = compute_corr_map(X_normalized, Y_DNb)

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)

# shared color scaling
vmin = -np.max(np.abs([corr_DAN, corr_DMN, corr_DNa, corr_DNb]))
vmax =  np.max(np.abs([corr_DAN, corr_DMN, corr_DNa, corr_DNb]))

def plot_heatmap(ax, corr, title, add_colorbar=False):
    sns.heatmap(
        corr, ax=ax,
        xticklabels=bands,
        yticklabels=lags,              # pretty labels
        cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax,
        cbar=add_colorbar,
        cbar_kws={'label': 'Correlation'} if add_colorbar else None
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Frequency band')
    ax.set_ylabel('Time bin')
    ax.set_xticks(np.arange(len(bands)) + 0.5)
    ax.set_xticklabels(bands, rotation=0)

# DAN
plot_heatmap(axes[0,0], corr_DAN, 'DAN')
# DMN
plot_heatmap(axes[0,1], corr_DMN, 'DMN')
# DNa
plot_heatmap(axes[1,0], corr_DNa, 'DNa')
# DNb (with colorbar)
plot_heatmap(axes[1,1], corr_DNb, 'DNb', add_colorbar=True)

plt.tight_layout()

# ---------- Save figure in Results folder ----------
results_dir = os.path.join(os.getcwd(), 'Results')
os.makedirs(results_dir, exist_ok=True)
fig_path = os.path.join(results_dir, 'correlation_maps.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {fig_path}")

plt.show()
