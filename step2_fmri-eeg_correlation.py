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
load_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025', 'processed_data', 'eeg_fmri_data.npz')
results_dir = os.path.join(os.getcwd(), 'Results')
os.makedirs(results_dir, exist_ok=True)

# ---- Statistical significance threshold (for FDR) ----
SIG_LEVEL = 0.01   # <- change to 0.01, 0.001, etc.

# ---- Plotting style for significance dots (visual only) ----
SHOW_SIG   = True
DOT_OPACITY = 0.65
DOT_SIZE    = 8
DOT_COLOR   = 'k'

# =========================
# Load data
# =========================
data = np.load(load_path, allow_pickle=True)

X = data['X']                     # (n_samples, n_channels, n_freqs, n_bins)
Y_DAN = data['Y_DAN']             # (n_samples, 1)
Y_DMN = data['Y_DMN']             # (n_samples, 1)
Y_DNa = data['Y_DNa']             # (n_samples, 1)
Y_DNb = data['Y_DNb']             # (n_samples, 1)
subject_ids = data['subject_ids'] # (n_samples,)
bin_times = data['bin_times']     # (n_bins,)

# Frequency axis or fallback
if 'freqs' in data.files:
    freqs = data['freqs'].astype(float)
else:
    freqs = np.linspace(1, 40, X.shape[2])

n_samples, n_channels, n_freqs, n_bins = X.shape
subjects = np.unique(subject_ids)

# =========================
# Axes / ticks
# =========================
xticks_hz_all = [0, 5, 10, 15, 20, 25, 30, 35, 40]
xticks_hz = [hz for hz in xticks_hz_all if (hz >= freqs.min()-1e-9 and hz <= freqs.max()+1e-9)]
xtick_positions = [int(np.argmin(np.abs(freqs - hz))) for hz in xticks_hz]

y_tick_indices = [i for i, t in enumerate(bin_times) if (int(t) == t and int(t) % 2 == 0)]
y_tick_labels  = [f"{int(bin_times[i])}s" for i in y_tick_indices]

# =========================
# Helpers
# =========================
def corr_map_for_subject(X_sub_norm, Y_sub):
    """Return (n_bins, n_freqs) correlation map for one subject."""
    fmri = Y_sub[:, 0]
    eeg_feat = X_sub_norm.mean(axis=1)  # (n_trials_sub, n_freqs, n_bins)
    corr = np.full((n_bins, n_freqs), np.nan, dtype=float)

    for f in range(n_freqs):
        x_ft = eeg_feat[:, f, :]  # (n_trials_sub, n_bins)
        for t in range(n_bins):
            x = x_ft[:, t]
            if np.std(x) == 0 or np.std(fmri) == 0 or np.sum(np.isfinite(x)) < 2:
                corr[t, f] = np.nan
            else:
                corr[t, f] = np.corrcoef(x, fmri)[0, 1]
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
    t_map = np.full((n_bins, n_freqs), np.nan, dtype=float)
    p_map = np.ones((n_bins, n_freqs), dtype=float)
    for i in range(n_bins):
        for j in range(n_freqs):
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

def overlay_significance_dots(ax, sig_mask, alpha=0.65, size=8, color='k'):
    """Overlay significant cells as dots (same orientation as heatmap)."""
    if not np.any(sig_mask):
        return
    ys, xs = np.where(sig_mask)
    ax.scatter(xs + 0.5, ys + 0.5, s=size, marker='o', linewidths=0,
               alpha=alpha, c=color)

# =========================
# Preprocess X: log10 then per-subject z-score (per channel × freq)
# =========================
epsilon = 1e-10
X_log = np.log10(X + epsilon)

r_DAN = np.zeros((len(subjects), n_bins, n_freqs), dtype=float)
r_DMN = np.zeros_like(r_DAN)
r_DNa = np.zeros_like(r_DAN)
r_DNb = np.zeros_like(r_DAN)

for si, subj in enumerate(subjects):
    idx = (subject_ids == subj)
    X_sub = X_log[idx]  # (n_trials_sub, n_channels, n_freqs, n_bins)

    X_sub_norm = np.empty_like(X_sub)
    n_trials = X_sub.shape[0]
    for ch in range(n_channels):
        for f in range(n_freqs):
            vals = X_sub[:, ch, f, :].reshape(n_trials, -1)
            vals_scaled = StandardScaler().fit_transform(vals)
            X_sub_norm[:, ch, f, :] = vals_scaled.reshape(n_trials, n_bins)

    r_DAN[si] = corr_map_for_subject(X_sub_norm, Y_DAN[idx])
    r_DMN[si] = corr_map_for_subject(X_sub_norm, Y_DMN[idx])
    r_DNa[si] = corr_map_for_subject(X_sub_norm, Y_DNa[idx])
    r_DNb[si] = corr_map_for_subject(X_sub_norm, Y_DNb[idx])

# =========================
# Save per-subject maps
# =========================
np.savez_compressed(
    os.path.join(results_dir, 'correlation_maps_per_subject.npz'),
    subjects=subjects, freqs=freqs, bin_times=bin_times,
    r_DAN=r_DAN, r_DMN=r_DMN, r_DNa=r_DNa, r_DNb=r_DNb
)
print("Saved per-subject maps to Results/correlation_maps_per_subject.npz")

# =========================
# Stats across subjects (two-sided one-sample t-test vs 0) + FDR
# =========================
stats_DAN = ttest_and_fdr(r_DAN, alpha=SIG_LEVEL)
stats_DMN = ttest_and_fdr(r_DMN, alpha=SIG_LEVEL)
stats_DNa = ttest_and_fdr(r_DNa, alpha=SIG_LEVEL)
stats_DNb = ttest_and_fdr(r_DNb, alpha=SIG_LEVEL)

np.savez_compressed(
    os.path.join(results_dir, 'correlation_maps_stats.npz'),
    bin_times=bin_times, freqs=freqs, subjects=subjects,
    DAN_t=stats_DAN["t"], DAN_p=stats_DAN["p"], DAN_p_fdr=stats_DAN["p_fdr"], DAN_sig=stats_DAN["sig"],
    DMN_t=stats_DMN["t"], DMN_p=stats_DMN["p"], DMN_p_fdr=stats_DMN["p_fdr"], DMN_sig=stats_DMN["sig"],
    DNa_t=stats_DNa["t"], DNa_p=stats_DNa["p"], DNa_p_fdr=stats_DNa["p_fdr"], DNa_sig=stats_DNa["sig"],
    DNb_t=stats_DNb["t"], DNb_p=stats_DNb["p"], DNb_p_fdr=stats_DNb["p_fdr"], DNb_sig=stats_DNb["sig"]
)
print("Saved stats to Results/correlation_maps_stats.npz")

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
# Plot with dedicated bottom colorbar row
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
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Lag (s)')
    ax.set_xlim(0, n_freqs)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([str(hz) for hz in xticks_hz], rotation=0)
    ax.set_ylim(n_bins, 0)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels(y_tick_labels, rotation=0)

    if SHOW_SIG:
        overlay_significance_dots(
            ax, sig,
            alpha=DOT_OPACITY,
            size=DOT_SIZE,
            color=DOT_COLOR
        )
    heatmaps.append(hm)

# Shared horizontal colorbar in bottom row
cax = fig.add_subplot(gs[2, :])
cbar = fig.colorbar(heatmaps[0].collections[0], cax=cax, orientation='horizontal')
cbar.set_label('Correlation')

plt.tight_layout()
fig_path = os.path.join(results_dir, "correlation_maps_group_mean_with_sig.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.show()
