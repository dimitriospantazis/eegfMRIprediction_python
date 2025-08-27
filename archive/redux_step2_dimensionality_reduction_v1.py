import os
import numpy as np

# -------- Paths --------
data_root = os.path.join(os.getcwd(), 'eegfmri_data_07122025', 'processed_data')
load_path = os.path.join(data_root, 'eeg_fmri_data.npz')
save_path = os.path.join(data_root, 'eeg_fmri_data_binned_2s_0to10s_canonicalbands.npz')

# -------- Load --------
data = np.load(load_path, allow_pickle=True)
X = data['X']                     # (n_samples, n_channels, n_freqs, n_bins)
Y_DAN = data['Y_DAN']
Y_DMN = data['Y_DMN']
Y_DNa = data['Y_DNa']
Y_DNb = data['Y_DNb']
subject_ids = data['subject_ids']
session_ids = data['session_ids']
bin_times = data['bin_times']     # e.g., [ 0, -1, -2, ..., -19 ]

n_samples, n_channels, n_freqs, n_bins = X.shape
epsilon = 1e-10

# If the frequency axis wasn't saved, assume the common 1–40 Hz grid used during extraction.
# Replace this with your exact freq vector if you saved it.
freqs = np.linspace(1, 40, n_freqs)

# -------- Log-transform (before aggregation across freqs) --------
# Yes—log first; then average across frequency within bands (geometric-mean behavior).
X_log = np.log10(X + epsilon)

# -------- Define 2-s time bins within 0–10 s (pre-event) --------
# Pair 1-s bins as [0,-1], [-2,-3], [-4,-5], [-6,-7], [-8,-9]
def idx_of(t):
    hits = np.where(bin_times == t)[0]
    if len(hits) == 0:
        raise ValueError(f"Required time bin {t}s not found in bin_times.")
    return int(hits[0])

time_groups = [
    [idx_of(0),  idx_of(-1)],   # 0–2 s before event
    [idx_of(-2), idx_of(-3)],   # 2–4 s
    [idx_of(-4), idx_of(-5)],   # 4–6 s
    [idx_of(-6), idx_of(-7)],   # 6–8 s
    [idx_of(-8), idx_of(-9)],   # 8–10 s
]

# Pretty labels & numeric edges
time_bin_edges_sec = np.array([(0,2), (2,4), (4,6), (6,8), (8,10)])       # numeric edges
time_bin_labels = [f"{a}\u2013{b}s" for (a,b) in time_bin_edges_sec]      # "0–2s", "2–4s", ...

# Aggregate (mean) across the 1-s bins inside each 2-s bin
# Result shape after time rebin: (n_samples, n_channels, n_freqs, 5)
X_time_binned = np.stack(
    [X_log[..., grp].mean(axis=-1) for grp in time_groups],
    axis=-1
)

# -------- Canonical frequency bands --------
band_defs = {
    "delta": (1, 4),    # 1–4 Hz
    "theta": (4, 8),    # 4–8 Hz
    "alpha": (8, 13),   # 8–13 Hz
    "beta":  (13, 30),  # 13–30 Hz
    "gamma": (30, 40),  # 30–40 Hz
}
band_names = list(band_defs.keys())
band_bounds = np.array([band_defs[b] for b in band_names])

# Build index lists (left-inclusive, right-exclusive, last band right-inclusive)
band_indices = []
for i, (lo, hi) in enumerate(band_bounds):
    if i < len(band_bounds) - 1:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
    else:
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    if idx.size == 0:
        raise ValueError(f"No frequency bins found for band {band_names[i]} [{lo},{hi}] Hz.")
    band_indices.append(idx)

# Aggregate (mean) within each band
# Output: (n_samples, n_channels, n_bands, 5)
X_binned = np.stack(
    [X_time_binned[:, :, idxs, :].mean(axis=2) for idxs in band_indices],
    axis=2
)

# -------- Clean printing --------
print(f"Original X shape: {X.shape}")
print(f"After time rebin (2s x 5): {X_time_binned.shape}")
print(f"Final X_binned shape (bands x time): {X_binned.shape}")
print("Bands: " + ", ".join(band_names))
print("Band bounds (Hz): " + ", ".join([f"[{lo},{hi}]" for (lo,hi) in band_bounds]))
print("Time bins: " + ", ".join(time_bin_labels))  # <- nice, readable

# -------- Save --------
np.savez_compressed(
    save_path,
    X_binned=X_binned,                  # (n_samples, n_channels, n_bands, 5)
    bands=np.array(band_names),
    band_bounds=band_bounds,            # [(lo,hi),...]
    time_bin_labels=np.array(time_bin_labels, dtype=object),
    time_bin_edges_sec=time_bin_edges_sec,   # numeric edges for programmatic use
    freqs=freqs,
    Y_DAN=Y_DAN,
    Y_DMN=Y_DMN,
    Y_DNa=Y_DNa,
    Y_DNb=Y_DNb,
    subject_ids=subject_ids,
    session_ids=session_ids
)
print(f"Saved to {save_path}")
