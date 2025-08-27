import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ---------- Load and Preprocess Data ----------
load_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025', 'processed_data', 'eeg_fmri_data.npz')
data = np.load(load_path, allow_pickle=True)

X = data['X']           # (n_samples, n_channels, n_freqs, n_bins)
Y_DAN = data['Y_DAN']   # (n_samples, 1)
Y_DMN = data['Y_DMN']
Y_DNa = data['Y_DNa']
Y_DNb = data['Y_DNb']
bin_times = data['bin_times']
subject_ids = data['subject_ids']

n_samples, n_channels, n_freqs, n_bins = X.shape
epsilon = 1e-10

# ---------- Log-transform ----------
X_log = np.log10(X + epsilon)

# ---------- Frequency axis (Hz) ----------
freqs = np.linspace(0, 40, n_freqs)  # 0..40 Hz inclusive

# ---------- Define canonical bands and average over them ----------
# (low, high) are half-open except the last band which includes the high edge.
band_defs = [
    ("Delta", 1.0, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 12.0),
    ("Beta", 13.0, 30.0),
    ("Gamma", 30.0, 40.1),  # include 40 Hz
]
band_names = [b[0] for b in band_defs]

band_idx_lists = []
for name, lo, hi in band_defs:
    if hi < 40.1:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
    else:
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    # If a band is empty due to coarse freq grid, keep it but avoid divide-by-zero later
    band_idx_lists.append(idx)

# Average X over freqs within each band -> shape: (n_samples, n_channels, n_bands, n_bins)
band_feats = []
for idx in band_idx_lists:
    if len(idx) == 0:
        band_feats.append(np.zeros((n_samples, n_channels, 1, n_bins), dtype=X_log.dtype))
    else:
        band_feats.append(np.mean(X_log[:, :, idx, :], axis=2, keepdims=True))
X_band = np.concatenate(band_feats, axis=2)  # (n_samples, n_channels, n_bands, n_bins)
n_bands = X_band.shape[2]

# ---------- Z-score normalize per channel & BAND ----------
X_normalized = np.zeros_like(X_band)
for ch in range(n_channels):
    for b in range(n_bands):
        vals = X_band[:, ch, b, :].reshape(n_samples, -1)
        scaler = StandardScaler()
        vals_scaled = scaler.fit_transform(vals)
        X_normalized[:, ch, b, :] = vals_scaled.reshape(n_samples, n_bins)

# ---------- Subject-wise correlation with Fisher-z ----------
def corr_map_by_subject(X_data, Y_data, subject_ids):
    uniq_subjects = np.unique(subject_ids)
    z_maps = []

    for subj in uniq_subjects:
        mask = (subject_ids == subj)
        if mask.sum() < 3:  # skip subjects with too few samples
            continue

        # Average EEG across channels -> (n_trials, n_bands, n_bins)
        eeg_feat = np.mean(X_data[mask], axis=1)
        y = Y_data[mask, 0]  # (n_trials,)

        # z-score y
        y_z = (y - np.mean(y)) / (np.std(y) + 1e-12)

        # z-score EEG per band/bin across trials
        eeg_z = (eeg_feat - np.mean(eeg_feat, axis=0, keepdims=True)) \
                / (np.std(eeg_feat, axis=0, keepdims=True) + 1e-12)

        # correlation = mean over trials of product of z-scores
        r_map = np.mean(eeg_z * y_z[:, None, None], axis=0)  # (n_bands, n_bins)

        # Fisher z-transform
        z_map = np.arctanh(np.clip(r_map, -0.999999, 0.999999))
        z_maps.append(z_map)

    if len(z_maps) == 0:
        return np.full((n_bins, n_bands), np.nan)

    # Average in z-space, then back to r
    z_avg = np.nanmean(np.stack(z_maps, axis=0), axis=0)  # (n_bands, n_bins)
    r_avg = np.tanh(z_avg)
    return r_avg.T  # (n_bins, n_bands)

# ---------- Compute group-level maps ----------
corr_DAN = corr_map_by_subject(X_normalized, Y_DAN, subject_ids)
corr_DMN = corr_map_by_subject(X_normalized, Y_DMN, subject_ids)
corr_DNa = corr_map_by_subject(X_normalized, Y_DNa, subject_ids)
corr_DNb = corr_map_by_subject(X_normalized, Y_DNb, subject_ids)

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
vmin = -np.max(np.abs([corr_DAN, corr_DMN, corr_DNa, corr_DNb]))
vmax =  np.max(np.abs([corr_DAN, corr_DMN, corr_DNa, corr_DNb]))

def plot_heatmap(ax, corr, title):
    hm = sns.heatmap(
        corr, ax=ax,
        xticklabels=band_names, yticklabels=False,
        cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax,
        cbar=False
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Frequency band')
    ax.set_ylabel('Time to event (s)')

    # Y ticks ~6 evenly spaced
    if corr.shape[0] > 6:
        y_idx = np.linspace(0, corr.shape[0]-1, 6, dtype=int)
    else:
        y_idx = np.arange(corr.shape[0])
    ax.set_yticks(y_idx)
    ax.set_yticklabels([f"{bin_times[i]:.1f}" for i in y_idx])
    return hm

plot_heatmap(axes[0,0], corr_DAN, 'DAN')
plot_heatmap(axes[0,1], corr_DMN, 'DMN')
plot_heatmap(axes[1,0], corr_DNa, 'DNa')
plot_heatmap(axes[1,1], corr_DNb, 'DNb')

# ---------- Add single horizontal colorbar ----------
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.03])
fig.colorbar(axes[0,0].collections[0], cax=cbar_ax, orientation='horizontal', label='Correlation')

plt.tight_layout(rect=[0, 0.08, 1, 1])

# ---------- Save ----------
results_dir = os.path.join(os.getcwd(), 'results')
os.makedirs(results_dir, exist_ok=True)
save_path = os.path.join(results_dir, 'eeg_fmri_correlation_maps_group_bands.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Group-level figure saved to: {save_path}")
