import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats

# =========================
# Config
# =========================
alpha_fdr = 0.05  # FDR q
results_dir = os.path.join(os.getcwd(), 'Results')
os.makedirs(results_dir, exist_ok=True)

# =========================
# Load ORIGINAL (unbinned)
# =========================
load_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025', 'processed_data', 'eeg_fmri_data.npz')
data = np.load(load_path, allow_pickle=True)

X = data['X']                     # (n_samples, n_channels, n_freqs, n_bins)
Y_dict = {
    'DAN': data['Y_DAN'],
    'DMN': data['Y_DMN'],
    'DNa': data['Y_DNa'],
    'DNb': data['Y_DNb'],
}
subject_ids = data['subject_ids']  # shape (n_samples,)
bin_times = data['bin_times']      # e.g., [0, -1, -2, ..., -19]

# Frequency axis or fallback
if 'freqs' in data.files:
    freqs = data['freqs'].astype(float)
else:
    freqs = np.linspace(1, 40, X.shape[2])

n_samples, n_channels, n_freqs, n_bins = X.shape
subjects = np.unique(subject_ids)
n_subj = len(subjects)
print(f"Subjects: {n_subj}, X: {X.shape}")

# =========================
# Within-subject log + z-score (per ch,f across trials)
# =========================
epsilon = 1e-10
X_log = np.log10(X + epsilon)

X_norm = np.zeros_like(X_log)
for s in subjects:
    m = (subject_ids == s)
    # per (ch,f), standardize columns (=time bins) across trials within subject
    for ch in range(n_channels):
        for f in range(n_freqs):
            vals = X_log[m, ch, f, :]                # shape (n_trials_for_s, n_bins)
            scaler = StandardScaler()
            X_norm[m, ch, f, :] = scaler.fit_transform(vals)

# =========================
# Helpers
# =========================
def per_subject_corr_maps(Xn, Y, subject_ids, average_channels=True):
    """
    Return r-maps per subject with shape (n_subj, n_bins, n_freqs).
    Correlates across trials within each subject.
    """
    maps = []
    subs_order = []
    for s in subjects:
        m = (subject_ids == s)
        if m.sum() < 3:  # need at least 3 points for corr
            maps.append(np.full((n_bins, n_freqs), np.nan))
            subs_order.append(s)
            continue
        fmri = Y[m, 0]  # (n_trials_s,)
        # features: (n_trials_s, n_freqs, n_bins)
        feat = Xn[m].mean(axis=1) if average_channels else Xn[m].reshape(m.sum(), n_channels, n_freqs, n_bins).mean(axis=1)
        # compute r for each (f,t)
        r = np.zeros((n_freqs, n_bins), dtype=float)
        for f in range(n_freqs):
            for t in range(n_bins):
                x = feat[:, f, t]
                if np.std(x) == 0 or np.std(fmri) == 0:
                    r[f, t] = np.nan
                else:
                    r[f, t] = np.corrcoef(x, fmri)[0, 1]
        maps.append(r.T)            # (n_bins, n_freqs)
        subs_order.append(s)
    return np.stack(maps, axis=0), np.array(subs_order)

def fdr_bh(pvals, q=0.05):
    """Benjamini–Hochberg FDR for any-shaped array; returns boolean mask of significant tests."""
    p = pvals.ravel()
    n = np.sum(~np.isnan(p))
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(p)+1)
    thresh = (ranks / max(n, 1)) * q
    is_sig = np.zeros_like(p, dtype=bool)
    valid = ~np.isnan(p)
    # largest k with p_k <= thresh_k
    passed = np.where(valid & (p <= thresh))[0]
    if passed.size:
        cutoff = np.max(passed)
        is_sig[order[:cutoff+1]] = True
    return is_sig.reshape(pvals.shape)

# =========================
# Compute per-subject maps and group stats
# =========================
results = {}
for name, Y in Y_dict.items():
    # r per subject
    r_maps, subs_order = per_subject_corr_maps(X_norm, Y, subject_ids, average_channels=True)  # (n_subj, n_bins, n_freqs)

    # Fisher z-transform for stats
    r_clip = np.clip(r_maps, -0.999999, 0.999999)
    z_maps = np.arctanh(r_clip)  # (n_subj, n_bins, n_freqs)

    # one-sample t-test vs 0 across subjects at each (t,f)
    t_stat, p_val = stats.ttest_1samp(z_maps, popmean=0.0, axis=0, nan_policy='omit')  # (n_bins, n_freqs)

    # FDR across all time×freq
    sig_mask = fdr_bh(p_val, q=alpha_fdr)



    # group mean r via inverse Fisher of mean z
    z_mean = np.nanmean(z_maps, axis=0)                      # (n_bins, n_freqs)
    r_group = np.tanh(z_mean)                                # back to r

    results[name] = dict(
        r_maps=r_maps, z_maps=z_maps, p_val=p_val,
        sig_mask=sig_mask, r_group=r_group, subjects=subs_order
    )
    # Save arrays
    np.savez_compressed(
        os.path.join(results_dir, f"group_corr_{name}.npz"),
        r_maps=r_maps, z_maps=z_maps, p_val=p_val, sig_mask=sig_mask,
        r_group=r_group, subjects=subs_order, freqs=freqs, bin_times=bin_times
    )

# =========================
# Axes (tick labels at BIN EDGES)
# =========================
xticks_hz = [0, 10, 20, 30, 40]
xtick_positions = [int(np.argmin(np.abs(freqs - hz))) for hz in xticks_hz]

# choose y ticks where bin_times is close to an even integer (handles negative floats)
y_tick_indices = [i for i, t in enumerate(bin_times) if abs(t - int(t)) < 1e-6 and int(t) % 2 == 0]
y_tick_labels  = [f"{int(bin_times[i])}s" for i in y_tick_indices]

# =========================
# Plot group mean r + FDR significant contours
# =========================
fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True, sharey=True)
order = ['DAN', 'DMN', 'DNa', 'DNb']

# common symmetric color scale across all networks
all_r = np.concatenate([results[k]['r_group'][None, ...] for k in order], axis=0)
vmax = np.nanmax(np.abs(all_r))
vmin = -vmax

def plot_panel(ax, r_group, sig_mask, title):
    hm = sns.heatmap(
        r_group, ax=ax, cmap='RdBu_r', center=0, vmin=vmin, vmax=vmax,
        cbar=False
    )
    # significance contour (True=significant)
    ax.contour(sig_mask, levels=[0.5], colors='k', linewidths=1.0)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Lag (s)')
    ax.set_xlim(0, n_freqs)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([str(hz) for hz in xticks_hz], rotation=0)
    ax.set_ylim(n_bins, 0)
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels(y_tick_labels, rotation=0)
    return hm

heatmaps = []
heatmaps.append(plot_panel(axes[0,0], results['DAN']['r_group'], results['DAN']['sig_mask'], 'DAN'))
heatmaps.append(plot_panel(axes[0,1], results['DMN']['r_group'], results['DMN']['sig_mask'], 'DMN'))
heatmaps.append(plot_panel(axes[1,0], results['DNa']['r_group'], results['DNa']['sig_mask'], 'DNa'))
heatmaps.append(plot_panel(axes[1,1], results['DNb']['r_group'], results['DNb']['sig_mask'], 'DNb'))

# shared horizontal colorbar
cbar_ax = fig.add_axes([0.25, -0.05, 0.5, 0.03])
cbar = fig.colorbar(heatmaps[0].collections[0], cax=cbar_ax, orientation='horizontal')
cbar.set_label('Group mean correlation (r)')

plt.tight_layout()
out_png = os.path.join(results_dir, 'group_corr_maps_with_FDR.png')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Figure saved to {out_png}")
plt.show()
