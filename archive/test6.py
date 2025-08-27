import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# =========================
# Paths & config
# =========================
root = os.path.join(os.getcwd(), "eegfmri_data_07122025")
bands_file = os.path.join(root, "processed_data", "eeg_fmri_data_binned_2s_0to10s_canonicalbands.npz")
target_key = "Y_DAN"       # or: "Y_DMN", "Y_DNa", "Y_DNb"
random_state = 1337

# Example EEG file (for channel positions / montage)
example_sub = "sub-001"
example_ses = "001"
example_bld = "001"
eegfile = os.path.join(
    root, example_sub, f"ses-{example_ses}", "eeg",
    f"{example_sub}_ses-{example_ses}_bld{example_bld}_eeg_Bergen_CWreg_filt_ICA_rej.set"
)

# XGBoost params
xgb_params = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.0, reg_lambda=1.0,
    objective="reg:squarederror", tree_method="hist",
    random_state=random_state, n_jobs=0
)

# =========================
# Load binned/band data
# =========================
Z = np.load(bands_file, allow_pickle=True)
X_binned = Z["X_binned"]                 # (n_samples, n_channels, n_bands, n_bins), log-aggregated
band_names = list(Z["bands"])            # e.g., ['delta','theta','alpha','beta','gamma']
subject_ids = Z["subject_ids"]
Y_all = Z[target_key][:, 0]

n_samples, n_channels, n_bands, n_bins = X_binned.shape
n_features = n_channels * n_bands * n_bins
print(f"Loaded X_binned: {X_binned.shape}  | bands={band_names}")

# =========================
# Subject-wise normalization (per prior pipeline)
# =========================
X_norm = np.zeros_like(X_binned)
for subj in np.unique(subject_ids):
    m = (subject_ids == subj)
    Xs = X_binned[m]
    Xs2d = Xs.reshape(Xs.shape[0], -1)
    scaler = StandardScaler()
    X_norm[m] = scaler.fit_transform(Xs2d).reshape(Xs.shape)

# Flatten for modeling
X_flat = X_norm.reshape(n_samples, -1)

# =========================
# Optional: LOSO evaluation
# =========================
logo = LeaveOneGroupOut()
fold_r2 = []
y_true_all, y_pred_all = [], []

for fold, (tr, te) in enumerate(logo.split(X_flat, Y_all, groups=subject_ids), start=1):
    X_tr, X_te = X_flat[tr], X_flat[te]
    y_tr, y_te = Y_all[tr], Y_all[te]

    scaler_fold = StandardScaler()
    X_trs = scaler_fold.fit_transform(X_tr)
    X_tes = scaler_fold.transform(X_te)

    model = XGBRegressor(**xgb_params)
    model.fit(X_trs, y_tr)

    y_hat = model.predict(X_tes)
    r2 = r2_score(y_te, y_hat)
    fold_r2.append(r2)
    y_true_all.append(y_te)
    y_pred_all.append(y_hat)
    print(f"Fold {fold:02d}  R2: {r2:.3f}")

if len(y_true_all) > 0:
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    overall_r2 = r2_score(y_true_all, y_pred_all)
    print(f"Overall LOSO R^2: {overall_r2:.3f}")

# =========================
# Fit on ALL data, get importances
# =========================
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_flat)

model_all = XGBRegressor(**xgb_params)
model_all.fit(X_all_scaled, Y_all)

gain = model_all.get_booster().get_score(importance_type="gain")  # {'f0': val, ...}
imp = np.zeros(n_features, dtype=float)
for k, v in gain.items():
    idx = int(k[1:])
    if idx < n_features:
        imp[idx] = v

# Reshape: (channels, bands, time) then collapse over time
imp_cbt = imp.reshape(n_channels, n_bands, n_bins)
imp_cb = imp_cbt.sum(axis=2)  # (n_channels, n_bands): per-channel, per-band importance

# =========================
# Load EEG once to get channel positions for topomap
# =========================
if not os.path.isfile(eegfile):
    raise FileNotFoundError(f"EEG file not found:\n{eegfile}\nUpdate example_sub/ses/bld near the top.")

raw = mne.io.read_raw_eeglab(eegfile, preload=False, verbose="ERROR")



# Ensure a montage exists; if absent, set a standard one (adjust if needed)
try:
    montage = raw.get_montage()
except Exception:
    montage = None

if montage is None:
    try:
        raw.set_montage("standard_1020")
        print("Applied standard_1020 montage (fallback).")
    except Exception as e:
        print("Warning: could not set a standard montage automatically:", e)

picks = mne.pick_types(raw.info, eeg=True, exclude="bads")

# Sanity check: channel count must match data
if len(picks) != n_channels:
    raise ValueError(f"Channel mismatch: EEG file has {len(picks)} EEG channels, but X has {n_channels}.")

# Use picked info directly with plot_topomap (positional second arg)
info_picked = mne.pick_info(raw.info, sel=picks)

# =========================
# Plot topographies: one map per band
# =========================
# Robust symmetric color scale across bands
vmax = np.percentile(np.abs(imp_cb), 99)
vmin = -vmax

fig, axes = plt.subplots(1, n_bands, figsize=(3.6 * n_bands, 3.8))
if n_bands == 1:
    axes = [axes]

images = []
for b, ax in enumerate(axes):
    data_band = imp_cb[:, b].astype(float)  # (n_channels,)

    # No vmin/vmax kwargs (compat); set clim on the returned image instead.
    im, _ = mne.viz.plot_topomap(
        data_band,
        info_picked,          # pass Info as 'pos' argument positionally
        axes=ax,
        cmap="RdBu_r",
        sensors=True,
        colorbar=False,       # shared colorbar below
        outlines="head",
        contours=0,
        show=False,
    )
    im.set_clim(vmin, vmax)   # enforce shared scale post-hoc
    images.append(im)
    ax.set_title(band_names[b], fontsize=12)

# Shared horizontal colorbar (outside)
cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.03])
norm = plt.cm.colors.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("XGBoost feature importance (gain, summed over time)")

plt.suptitle(f"Feature importance per channel (summed over time), target={target_key}", y=0.98)
plt.tight_layout(rect=[0, 0.1, 1, 0.95])

# Save
results_dir = os.path.join(os.getcwd(), "Results")
os.makedirs(results_dir, exist_ok=True)
fig_path = os.path.join(results_dir, f"topomap_feature_importance_{target_key}.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Topomap figure saved to {fig_path}")
plt.show()
