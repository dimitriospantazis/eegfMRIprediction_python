import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp
import matplotlib.gridspec as gridspec

# =========================
# Config / Paths
# =========================
data_root = os.path.join(os.getcwd(), 'eegfmri_data_07122025', 'processed_data')
load_path = os.path.join(data_root, 'eeg_fmri_data_binned_2s_0to20s_canonicalbands.npz')
results_dir = os.path.join(os.getcwd(), 'Results')
os.makedirs(results_dir, exist_ok=True)

# ---- Statistical significance threshold (for FDR) ----
SIG_LEVEL = 0.01   # change to 0.05, 0.001, etc.

# ---- Plotting style for significance dots (visual only) ----
SHOW_SIG    = True
DOT_OPACITY = 0.65
DOT_SIZE    = 10
DOT_COLOR   = 'k'

# =========================
# Load binned data
# =========================
data = np.load(load_path, allow_pickle=True)

X_binned = data['X_binned']         # (n_samples, n_channels, n_bands, n_bins)
bands    = data['bands']            # array of band names (strings)
time_bin_edges_sec = data['time_bin_edges_sec']  # shape (n_bins, 2) numeric edges

Y_DAN = data['Y_DAN']
Y_DMN = data['Y_DMN']
Y_DNa = data['Y_DNa']
Y_DNb = data['Y_DNb']
subject_ids = data['subject_ids']

n_samples, n_channels, n_bands, n_bins = X_binned.shape
subjects = np.unique(subject_ids)

# =========================
# Axes / ticks
# =========================
# X (bands): ticks at centers (0.5, 1.5, ...)
x_centers = np.arange(n_bands) + 0.5
x_labels  = [str(b) for b in bands]

# Y (time): ticks at **edges** (0..n_bins), labels = 0s, 2s, ..., 20s
left_edges = time_bin_edges_sec[:, 0].astype(int)  # e.g., [0,2,...,18]
final_right_edge = int(time_bin_edges_sec[-1, 1])  # e.g., 20
edge_values = np.concatenate([left_edges, [final_right_edge]])  # [0,2,...,20]
y_edges = np.arange(n_bins + 1)  # 0..n_bins (edges of rows)
y_edge_labels = [f"{t}s" for t in edge_values]

# =========================
# Helpers
# =========================
def corr_map_for_subject(X_sub_norm, Y_sub):
    """
    X_sub_norm: (n_trials_sub, n_channels, n_bands, n_bins)  normalized within subject
    Y_sub:      (n_trials_sub, 1)
    Returns:    (n_bins, n_bands)   # time × band
    """
    fmri = Y_sub[:, 0]
    eeg_feat = X_sub_norm.mean(axis=1)  # -> (n_trials_sub, n_bands, n_bins)

    corr = np.full((n_bins, n_bands), np.nan, dtype=float)
    for b in range(n_bands):
        x_b = eeg_feat[:, b, :]
        for t in range(n_bins):
            x = x_b[:, t]
            if np.std(x) == 0 or np.std(fmri) == 0 or np.sum(np.isfinite(x)) < 2:
                corr[t, b] = np.nan
            else:
                corr[t, b] = np.corrcoef(x, fmri)[0, 1]
    return corr

def fdr_bh(pvals, alpha=0.05):
    """Benjamini–Hochberg FDR; returns (p_adj, sig_mask) same shape as pvals."""
    p = pvals.ravel().copy()
    n = p.size
    order = np.argsort(p)
    ranked = np.arange(1, n + 1)
    p_sorted = p[order]
    p_adj_sorted = np.minimum.accumulate((p_sorted * n / ranked)[::-1])[::-1]
    p_adj = np.empty_like(p)
    p_adj[order] = np.clip(p_adj_sorted, 0, 1)
    thresh = alpha * ranked / n
    passed = p_sorted <= thresh
    sig = np.zeros_like(p, dtype=bool)
    if np.any(passed):
        crit_idx = np.max(np.where(passed)[0])
        sig[order[:crit_idx + 1]] = True
    return p_adj.reshape(pvals.shape), sig.reshape(pvals.shape)

def ttest_and_fdr(r_stack, alpha=0.05):
    """Two-sided one-sample t-test across subjects vs 0; FDR correct."""
    t_map = np.full((n_bins, n_bands), np.nan, dtype=float)
    p_map = np.ones((n_bins, n_bands), dtype=float)
    for i in range(n_bins):
        for j in range(n_bands):
            vals = r_stack[:, i, j]
            vals = vals[np.isfinite(vals)]
            if vals.size >= 2 and np.std(vals) > 0:
                t, p = ttest_1samp(vals, 0.0, alternative='two-sided', nan_policy='omit')
                t_map[i, j] = t
                p_map[i, j] = p
            else:
                t_map[i, j] = np.nan
                p_map[i, j] = 1.0
    p_fdr, sig = fdr_bh(p_map, alpha=alpha)
    return {"t": t_map, "p": p_map, "p_fdr": p_fdr, "sig": sig}

def overlay_significance_dots(ax, sig_mask, alpha=0.65, size=10, color='k'):
    """Overlay significant cells as dots (center of each cell)."""
    if not np.any(sig_mask):
        return
    ys, xs = np.where(sig_mask)  # y=time index, x=band index
    ax.scatter(xs + 0.5, ys + 0.5, s=size, marker='o', linewidths=0,
               alpha=alpha, c=color, zorder=3)

# =========================
# Per-subject normalization (z-score per channel×band within subject)
# =========================
# NOTE: X_binned already logged during binning; don't log again.
r_DAN = np.zeros((len(subjects), n_bins, n_bands), dtype=float)
r_DMN = np.zeros_like(r_DAN)
r_DNa = np.zeros_like(r_DAN)
r_DNb = np.zeros_like(r_DAN)

for si, subj in enumerate(subjects):
    idx = (subject_ids == subj)
    X_sub = X_binned[idx]  # (n_trials_sub, n_channels, n_bands, n_bins)

    X_sub_norm = np.empty_like(X_sub)
    n_trials = X_sub.shape[0]
    for ch in range(n_channels):
        for b in range(n_bands):
            vals = X_sub[:, ch, b, :].reshape(n_trials, -1)
            vals_scaled = StandardScaler().fit_transform(vals)
            X_sub_norm[:, ch, b, :] = vals_scaled.reshape(n_trials, n_bins)

    r_DAN[si] = corr_map_for_subject(X_sub_norm, Y_DAN[idx])
    r_DMN[si] = corr_map_for_subject(X_sub_norm, Y_DMN[idx])
    r_DNa[si] = corr_map_for_subject(X_sub_norm, Y_DNa[idx])
    r_DNb[si] = corr_map_for_subject(X_sub_norm, Y_DNb[idx])

# =========================
# Save per-subject maps
# =========================
np.savez_compressed(
    os.path.join(results_dir, 'correlation_maps_per_subject_BANDS_0to20s_edges.npz'),
    subjects=subjects, bands=bands, time_bin_edges_sec=time_bin_edges_sec,
    r_DAN=r_DAN, r_DMN=r_DMN, r_DNa=r_DNa, r_DNb=r_DNb
)
print("Saved per-subject maps to Results/correlation_maps_per_subject_BANDS_0to20s_edges.npz")

# =========================
# Stats across subjects (two-sided one-sample t-test vs 0) + FDR
# =========================
stats_DAN = ttest_and_fdr(r_DAN, alpha=SIG_LEVEL)
stats_DMN = ttest_and_fdr(r_DMN, alpha=SIG_LEVEL)
stats_DNa = ttest_and_fdr(r_DNa, alpha=SIG_LEVEL)
stats_DNb = ttest_and_fdr(r_DNb, alpha=SIG_LEVEL)

np.savez_compressed(
    os.path.join(results_dir, 'correlation_maps_stats_BANDS_0to20s_edges.npz'),
    time_bin_edges_sec=time_bin_edges_sec, bands=bands, subjects=subjects,
    DAN_t=stats_DAN["t"], DAN_p=stats_DAN["p"], DAN_p_fdr=stats_DAN["p_fdr"], DAN_sig=stats_DAN["sig"],
    DMN_t=stats_DMN["t"], DMN_p=stats_DMN["p"], DMN_p_fdr=stats_DMN["p_fdr"], DMN_sig=stats_DMN["sig"],
    DNa_t=stats_DNa["t"], DNa_p=stats_DNa["p"], DNa_p_fdr=stats_DNa["p_fdr"], DNa_sig=stats_DNa["sig"],
    DNb_t=stats_DNb["t"], DNb_p=stats_DNb["p"], DNb_p_fdr=stats_DNb["p_fdr"], DNb_sig=stats_DNb["sig"]
)
print("Saved stats to Results/correlation_maps_stats_BANDS_0to20s_edges.npz")

# =========================
# Group means for plotting
# =========================
r_mean = {
    'DAN': np.nanmean(r_DAN, axis=0),
    'DMN': np.nanmean(r_DMN, axis=0),
    'DNa': np.nanmean(r_DNa, axis=0),
    'DNb': np.nanmean(r_DNb, axis=0),
}
titles = ['DAN', 'DMN', 'DNa', 'DNb']
mats   = [r_mean['DAN'], r_mean['DMN'], r_mean['DNa'], r_mean['DNb']]
sigs   = [stats_DAN["sig"], stats_DMN["sig"], stats_DNa["sig"], stats_DNb["sig"]]

vmax = np.nanmax(np.abs(mats))
vmin = -vmax

# =========================
# Plot with band-centered x-ticks and edge-based y-ticks
# =========================
fig = plt.figure(figsize=(20, 11))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.05])  # bottom row for colorbar

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
]

heatmaps = []
for ax, mat, title, sig in zip(axes, mats, titles, sigs):
    hm = sns.heatmap(
        mat, ax=ax, cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax, cbar=False
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Time (s)')

    # Cell geometry: columns [0..n_bands], rows [0..n_bins]
    ax.set_xlim(0, n_bands)
    ax.set_ylim(n_bins, 0)  # invert y so 0s at top

    # X ticks at band centers; Y ticks at bin edges (0..n_bins)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(x_labels, rotation=0)

    ax.set_yticks(y_edges)
    ax.set_yticklabels(y_edge_labels, rotation=0)

    if SHOW_SIG:
        overlay_significance_dots(ax, sig, alpha=DOT_OPACITY, size=DOT_SIZE, color=DOT_COLOR)

    heatmaps.append(hm)

# Shared horizontal colorbar in bottom row
cax = fig.add_subplot(gs[2, :])
cbar = fig.colorbar(heatmaps[0].collections[0], cax=cax, orientation='horizontal')
cbar.set_label('Correlation')

plt.tight_layout()
fig_path = os.path.join(results_dir, "correlation_maps_group_mean_BANDS_0to20s.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.show()
