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
subject_ids = data['subject_ids']  # length n_samples (one id per sample)
bin_times = data['bin_times']     # e.g., [0, -1, -2, ..., -19]

# Frequency axis or fallback
if 'freqs' in data.files:
    freqs = data['freqs'].astype(float)
else:
    freqs = np.linspace(1, 40, X.shape[2])

n_samples, n_channels, n_freqs, n_bins = X.shape
subjects = np.unique(subject_ids)

# ---------- Axes (tick labels at BIN EDGES) ----------
# X ticks: nice round Hz
xticks_hz = [0, 10, 20, 30, 40]
xtick_positions = [int(np.argmin(np.abs(freqs - hz))) for hz in xticks_hz]

# Y ticks: every 2 s
y_tick_indices = [i for i, t in enumerate(bin_times) if t % 2 == 0]
y_tick_labels  = [f"{int(t)}s" for t in bin_times[y_tick_indices]]

# ---------- Helper to compute one subject's corr map for one target ----------
def corr_map_for_subject(X_sub_norm, Y_sub):
    """
    X_sub_norm: (n_trials_subj, n_channels, n_freqs, n_bins) normalized within subject
    Y_sub:      (n_trials_subj, 1)
    Returns:    (n_bins, n_freqs)
    """
    fmri = Y_sub[:, 0]  # (n_trials_subj,)
    eeg_feat = X_sub_norm.mean(axis=1)  # average across channels -> (n_trials_subj, n_freqs, n_bins)

    # Compute correlation per (f, t)
    corr = np.zeros((n_bins, n_freqs), dtype=float)
    for f in range(n_freqs):
        x_ft = eeg_feat[:, f, :]  # (n_trials_subj, n_bins)
        for t in range(n_bins):
            x = x_ft[:, t]
            if np.std(x) == 0 or np.std(fmri) == 0:
                corr[t, f] = np.nan
            else:
                corr[t, f] = np.corrcoef(x, fmri)[0, 1]
    return corr

# ---------- Subject-wise normalization (within subject) and correlation ----------
epsilon = 1e-10
X_log = np.log10(X + epsilon)

# containers: (n_subjects, n_bins, n_freqs)
r_DAN = np.zeros((len(subjects), n_bins, n_freqs), dtype=float)
r_DMN = np.zeros_like(r_DAN)
r_DNa = np.zeros_like(r_DAN)
r_DNb = np.zeros_like(r_DAN)

for si, subj in enumerate(subjects):
    idx = (subject_ids == subj)
    X_sub = X_log[idx]                     # (n_trials, n_channels, n_freqs, n_bins)
    # Normalize per (channel, freq) within this subject
    X_sub_norm = np.empty_like(X_sub)
    n_trials = X_sub.shape[0]
    for ch in range(n_channels):
        for f in range(n_freqs):
            vals = X_sub[:, ch, f, :].reshape(n_trials, -1)  # (n_trials, n_bins)
            vals_scaled = StandardScaler().fit_transform(vals)
            X_sub_norm[:, ch, f, :] = vals_scaled.reshape(n_trials, n_bins)

    # Compute maps for each target for this subject
    r_DAN[si] = corr_map_for_subject(X_sub_norm, Y_DAN[idx])
    r_DMN[si] = corr_map_for_subject(X_sub_norm, Y_DMN[idx])
    r_DNa[si] = corr_map_for_subject(X_sub_norm, Y_DNa[idx])
    r_DNb[si] = corr_map_for_subject(X_sub_norm, Y_DNb[idx])

# ---------- Save per-subject maps ----------
results_dir = os.path.join(os.getcwd(), 'Results')
os.makedirs(results_dir, exist_ok=True)
np.savez_compressed(
    os.path.join(results_dir, 'correlation_maps_per_subject.npz'),
    subjects=subjects, freqs=freqs, bin_times=bin_times,
    r_DAN=r_DAN, r_DMN=r_DMN, r_DNa=r_DNa, r_DNb=r_DNb
)
print(f"Saved per-subject maps to {os.path.join(results_dir, 'correlation_maps_per_subject.npz')}")

import matplotlib.gridspec as gridspec

# ---------- Group mean plot (quick sanity check; still per-subject computed) ----------
r_mean = {
    'DAN': np.nanmean(r_DAN, axis=0),
    'DMN': np.nanmean(r_DMN, axis=0),
    'DNa': np.nanmean(r_DNa, axis=0),
    'DNb': np.nanmean(r_DNb, axis=0),
}

titles = ['DAN', 'DMN', 'DNa', 'DNb']
mats   = [r_mean['DAN'], r_mean['DMN'], r_mean['DNa'], r_mean['DNb']]

vmax = np.nanmax(np.abs(mats))
vmin = -vmax

# ---------- Figure layout with dedicated row for colorbar ----------
fig = plt.figure(figsize=(20, 11))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.05])  # last row = colorbar

axes = [
    fig.add_subplot(gs[0,0]),
    fig.add_subplot(gs[0,1]),
    fig.add_subplot(gs[1,0]),
    fig.add_subplot(gs[1,1]),
]

heatmaps = []
for ax, mat, title in zip(axes, mats, titles):
    hm = sns.heatmap(
        mat, ax=ax, cmap="RdBu_r", center=0, vmin=vmin, vmax=vmax, cbar=False
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Lag (s)")
    ax.set_xlim(0, n_freqs)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([str(hz) for hz in xticks_hz], rotation=0)
    ax.set_ylim(n_bins, 0)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels(y_tick_labels, rotation=0)
    heatmaps.append(hm)

# ---------- Shared horizontal colorbar in bottom row ----------
cax = fig.add_subplot(gs[2, :])  # full bottom row
cbar = fig.colorbar(
    heatmaps[0].collections[0], cax=cax, orientation="horizontal"
)
cbar.set_label("Correlation")

plt.tight_layout()
fig_path = os.path.join(results_dir, "correlation_maps_group_mean.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to {fig_path}")
plt.show()


