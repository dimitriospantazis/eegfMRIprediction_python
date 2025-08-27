import numpy as np
import os

# ---------- Load ----------
load_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025', 'processed_data', 'eeg_fmri_data.npz')
data = np.load(load_path, allow_pickle=True)

X = data['X']           # (n_samples, n_channels, n_freqs, n_bins)
bin_times = data['bin_times']  # e.g., [0, -1, -2, ..., -19]
subject_ids = data['subject_ids']
Y_DAN = data['Y_DAN']
Y_DMN = data['Y_DMN']
Y_DNa = data['Y_DNa']
Y_DNb = data['Y_DNb']
Y = data['Y']  # (n_samples, n_targets)

n_samples, n_channels, n_freqs, n_bins = X.shape
epsilon = 1e-10

# ---------- Log-transform ----------
X_log = np.log10(X + epsilon)

# ---------- Frequency axis ----------
freqs = np.linspace(0, 40, n_freqs)

# ---------- Canonical bands ----------
band_defs = [
    ("Delta", 1.0, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 12.0),
    ("Beta", 13.0, 30.0),
    ("Gamma", 30.0, 40.1),  # include 40 Hz
]
band_names = [b[0] for b in band_defs]

# Map freqs -> band indices
band_idx_lists = []
for _, lo, hi in band_defs:
    if hi < 40.1:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
    else:
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    band_idx_lists.append(idx)

# ---------- Average over canonical bands ----------
band_feats = []
for idx in band_idx_lists:
    if len(idx) == 0:
        band_feats.append(np.zeros((n_samples, n_channels, 1, n_bins), dtype=X_log.dtype))
    else:
        band_feats.append(np.mean(X_log[:, :, idx, :], axis=2, keepdims=True))
X_band = np.concatenate(band_feats, axis=2)  # (n_samples, n_channels, n_bands, n_bins)
n_bands = X_band.shape[2]

# ---------- Keep only bins in [0, -10] seconds ----------
mask_0_to_10 = (bin_times <= 0.0) & (bin_times >= -10.0)
keep_idx = np.where(mask_0_to_10)[0]          # indices in original ordering (0, -1, -2, ...)
bin_times_kept = bin_times[keep_idx]          # length = 11 (0..-10)

# ---------- Downsample to 2 s: average adjacent bins ----------
# We will pair (0,-1), (-2,-3), (-4,-5), (-6,-7), (-8,-9). The lone -10 is dropped to keep strict 2 s windows.
if len(keep_idx) % 2 == 1:
    # Drop the last (earliest) bin to make an even count
    keep_idx = keep_idx[:-1]
    bin_times_kept = bin_times_kept[:-1]

# Create pairs of indices
keep_idx = keep_idx.reshape(-1, 2)            # shape: (n_pairs, 2)

# Average the data across each pair on the last axis (time bins)
# Result shape: (n_samples, n_channels, n_bands, n_pairs)
X_band_t2 = np.stack([np.mean(X_band[:, :, :, pair], axis=3) for pair in keep_idx], axis=3)

# New bin_times are the mean of each 2 s pair (e.g., (0,-1)->-0.5, (-2,-3)->-2.5, ...)
new_bin_times = np.array([bin_times[pair].mean() for pair in keep_idx])

print(f"Original bins: {bin_times}")
print(f"Kept (0..-10s): {bin_times[mask_0_to_10]}")
print(f"Downsampled to 2s, new times: {new_bin_times}")
print(f"X_band_t2 shape: {X_band_t2.shape}  (samples, channels, bands, new_bins)")

# ---------- Save to disk ----------
save_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025', 'processed_data', 'eegfmri_data_bands_t2s.npz')
np.savez_compressed(
    save_path,
    X_band=X_band_t2,
    band_names=np.array(band_names),
    bin_times=new_bin_times,
    subject_ids=subject_ids,
    Y_DAN=Y_DAN,
    Y_DMN=Y_DMN,
    Y_DNa=Y_DNa,
    Y_DNb=Y_DNb,
    Y=Y
)
print(f"Saved band & 2s-time-averaged data to: {save_path}")
