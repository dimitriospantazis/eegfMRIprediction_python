import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# ---------- Load ORIGINAL (unbinned) Data ----------
load_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025', 'processed_data', 'eeg_fmri_data.npz')
data = np.load(load_path, allow_pickle=True)

X = data['X']                     # (n_samples, n_channels, n_freqs, n_bins)
Y_DAN = data['Y_DAN']
Y_DMN = data['Y_DMN']
Y_DNa = data['Y_DNa']
Y_DNb = data['Y_DNb']
subject_ids = data['subject_ids']
bin_times = data['bin_times']     # e.g., [0, -1, -2, ..., -19]

# Frequency axis or fallback
if 'freqs' in data.files:
    freqs = data['freqs'].astype(float)
else:
    freqs = np.linspace(1, 40, X.shape[2])

n_samples, n_channels, n_freqs, n_bins = X.shape

# ---------- Log-transform then Z-score normalize per channel & frequency ----------
epsilon = 1e-10
X_log = np.log10(X + epsilon)

X_normalized = np.zeros_like(X_log)
for ch in range(n_channels):
    for f in range(n_freqs):
        vals = X_log[:, ch, f, :].reshape(n_samples, -1)
        scaler = StandardScaler()
        vals_scaled = scaler.fit_transform(vals)
        X_normalized[:, ch, f, :] = vals_scaled.reshape(n_samples, n_bins)

# ---------- Axes (tick labels at BIN EDGES) ----------
# X ticks: nice round Hz
xticks_hz = [0, 10, 20, 30, 40]
xtick_positions = [int(np.argmin(np.abs(freqs - hz))) for hz in xticks_hz]

# Y ticks: every 2 s
y_tick_indices = [i for i, t in enumerate(bin_times) if t % 2 == 0]
y_tick_labels  = [f"{int(t)}s" for t in bin_times[y_tick_indices]]

# ---------- Function: compute correlation map ----------
def compute_corr_map_unbinned(X_data, Y_data):
    fmri_signal = Y_data[:, 0]
    eeg_features = np.mean(X_data, axis=1)  # average across channels
    corr_map = np.zeros((n_freqs, n_bins), dtype=float)
    for f in range(n_freqs):
        for t in range(n_bins):
            corr_map[f, t] = np.corrcoef(eeg_features[:, f, t], fmri_signal)[0, 1]
    return corr_map.T  # (n_bins, n_freqs)

# ---------- Compute correlation maps ----------
corr_DAN = compute_corr_map_unbinned(X_normalized, Y_DAN)
corr_DMN = compute_corr_map_unbinned(X_normalized, Y_DMN)
corr_DNa = compute_corr_map_unbinned(X_normalized, Y_DNa)
corr_DNb = compute_corr_map_unbinned(X_normalized, Y_DNb)

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)

vmax = np.max(np.abs([corr_DAN, corr_DMN, corr_DNa, corr_DNb]))
vmin = -vmax

# We'll keep track of heatmap handles for shared colorbar
heatmaps = []

def plot_heatmap(ax, corr, title):
    hm = sns.heatmap(
        corr, ax=ax,
        cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax,
        cbar=False
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Lag (s)')
    ax.set_xlim(0, n_freqs)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([str(hz) for hz in xticks_hz], rotation=0)
    ax.set_ylim(n_bins, 0)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels(y_tick_labels, rotation=0)
    heatmaps.append(hm)

plot_heatmap(axes[0,0], corr_DAN, 'DAN')
plot_heatmap(axes[0,1], corr_DMN, 'DMN')
plot_heatmap(axes[1,0], corr_DNa, 'DNa')
plot_heatmap(axes[1,1], corr_DNb, 'DNb')

# ---------- Shared horizontal colorbar ----------
cbar_ax = fig.add_axes([0.25, -0.05, 0.5, 0.03])  # [left, bottom, width, height]
cbar = fig.colorbar(heatmaps[0].collections[0], cax=cbar_ax, orientation='horizontal')
cbar.set_label('Correlation')

plt.tight_layout()

# ---------- Save figure ----------
results_dir = os.path.join(os.getcwd(), 'Results')
os.makedirs(results_dir, exist_ok=True)
fig_path = os.path.join(results_dir, 'correlation_maps_unbinned.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {fig_path}")

plt.show()
