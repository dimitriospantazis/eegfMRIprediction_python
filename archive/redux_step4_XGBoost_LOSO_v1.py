import os
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import shap  # pip install shap

# =========================
# Paths & config
# =========================
data_root = os.path.join(os.getcwd(), "eegfmri_data_07122025")
bands_file = os.path.join(
    data_root, "processed_data",
    "eeg_fmri_data_binned_2s_0to10s_canonicalbands.npz"
)
results_dir = os.path.join(data_root, "results")
os.makedirs(results_dir, exist_ok=True)

target_key = "Y_DAN"        # or "Y_DMN", "Y_DNa", "Y_DNb"
random_state = 1337

xgb_params = dict(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=random_state,
    n_jobs=0
)

# =========================
# Load
# =========================
Z = np.load(bands_file, allow_pickle=True)

X_band = Z["X_binned"]                               # (n_samples, n_channels, n_bands, n_bins)
band_names = list(Z["bands"])                        # ['delta','theta','alpha','beta','gamma']

# Bin centers (negative, pre-event)
if "time_bin_edges_sec" in Z.files:
    edges = Z["time_bin_edges_sec"].astype(float)    # shape (5,2) like [(0,2),(2,4),...]
    bin_times = -edges.mean(axis=1)                  # e.g., [-1,-3,-5,-7,-9]
else:
    n_bins_fallback = X_band.shape[-1]
    bin_times = -(np.arange(n_bins_fallback) * 2 + 1).astype(float)

subject_ids = Z["subject_ids"]
Y_all = Z[target_key][:, 0]

n_samples, n_channels, n_bands, n_bins = X_band.shape
n_features = n_channels * n_bands * n_bins
print(f"Loaded X_binned: {X_band.shape} — subjects: {len(np.unique(subject_ids))}")

# =========================
# Subject-wise normalization
# =========================
X_norm = np.zeros_like(X_band)
for subj in np.unique(subject_ids):
    m = (subject_ids == subj)
    Xs2d = X_band[m].reshape(np.sum(m), -1)
    scaler_subj = StandardScaler()
    X_norm[m] = scaler_subj.fit_transform(Xs2d).reshape(np.sum(m), n_channels, n_bands, n_bins)

X_flat = X_norm.reshape(n_samples, -1)

# Helpers
def index_to_cbb(idx):
    ch, rem = divmod(idx, n_bands * n_bins)
    b, t = divmod(rem, n_bins)
    return ch, b, t

def feature_name(idx):
    ch, b, t = index_to_cbb(idx)
    return f"ch{ch:02d}|{band_names[b]}|t={bin_times[t]:.1f}s"

# =========================
# LOSO with per-fold scaling (unchanged)
# =========================
logo = LeaveOneGroupOut()
y_true_all, y_pred_all, fold_r2 = [], [], []

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
    print(f"Fold {fold:02d}  R^2: {r2:.3f}")

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)
overall_r2 = r2_score(y_true_all, y_pred_all)
print(f"Overall LOSO R^2: {overall_r2:.3f}")

# =========================
# Fit on ALL data and compute SHAP (signed + abs)
# =========================
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_flat)

model_all = XGBRegressor(**xgb_params)
model_all.fit(X_all_scaled, Y_all)

# TreeSHAP
try:
    explainer = shap.Explainer(model_all)   # uses TreeSHAP internally for XGB
    sv = explainer(X_all_scaled)            # shap.Explanation
    shap_values = sv.values                 # (n_samples, n_features)
except Exception:
    explainer = shap.TreeExplainer(model_all)
    shap_values = explainer.shap_values(X_all_scaled)

shap_values = np.asarray(shap_values)       # ensure ndarray

# Global metrics
shap_mean_abs    = np.abs(shap_values).mean(axis=0)   # (n_features,) magnitude
shap_mean_signed = shap_values.mean(axis=0)           # (n_features,) signed direction

# Reshape & aggregate
imp_abs_cbb    = shap_mean_abs.reshape(n_channels, n_bands, n_bins)
imp_signed_cbb = shap_mean_signed.reshape(n_channels, n_bands, n_bins)

imp_abs_per_channel    = imp_abs_cbb.sum(axis=(1, 2))      # (ch,)
imp_signed_per_channel = imp_signed_cbb.sum(axis=(1, 2))   # (ch,)

band_time_abs    = imp_abs_cbb.sum(axis=0)                 # (band, time)
band_time_signed = imp_signed_cbb.sum(axis=0)              # (band, time)

chan_time_abs    = imp_abs_cbb.sum(axis=1)                 # (ch, time)
chan_time_signed = imp_signed_cbb.sum(axis=1)              # (ch, time)

# Sanity printouts
top_k = 20
top_idx_abs = np.argsort(shap_mean_abs)[::-1][:top_k]
print("\nTop features by mean |SHAP|:")
for r, idx in enumerate(top_idx_abs, 1):
    print(f"{r:2d}. {feature_name(idx)}   |SHAP|={shap_mean_abs[idx]:.4g}  signed={shap_mean_signed[idx]:+.3g}")

print("\nBand × Time (sum mean|SHAP| over channels):")
for b, bname in enumerate(band_names):
    row = ", ".join([f"{bin_times[t]:.1f}s:{band_time_abs[b, t]:.3g}" for t in range(n_bins)])
    print(f"{bname:>6}: {row}")

# =========================
# Save artifacts (abs + signed)
# =========================
out_file = os.path.join(results_dir, f"xgb_importance_SHAP_{target_key}.npz")
np.savez_compressed(
    out_file,
    target_key=target_key,
    # per-feature
    shap_mean_abs=shap_mean_abs,                  # (n_features,)
    shap_mean_signed=shap_mean_signed,            # (n_features,)
    # reshaped (ch, band, time)
    imp_abs_cbb=imp_abs_cbb,
    imp_signed_cbb=imp_signed_cbb,
    # per-channel (sum over band×time)
    imp_abs_per_channel=imp_abs_per_channel,          # (n_channels,)
    imp_signed_per_channel=imp_signed_per_channel,    # (n_channels,)
    # summaries
    band_time_abs=band_time_abs,                  # (n_bands, n_bins)
    band_time_signed=band_time_signed,            # (n_bands, n_bins)
    chan_time_abs=chan_time_abs,                  # (n_channels, n_bins)
    chan_time_signed=chan_time_signed,            # (n_channels, n_bins)
    # meta
    band_names=np.array(band_names),
    bin_times=np.array(bin_times, dtype=float),
    n_channels=n_channels,
    n_bands=n_bands,
    n_bins=n_bins,
    overall_r2=overall_r2,
    fold_r2=np.array(fold_r2, dtype=float),
    top_idx_abs=top_idx_abs
)
print(f"\nSaved SHAP importance (abs + signed) to: {out_file}")






data_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025')
example_sub = "sub-001"
example_ses = "001"
example_bld = "001"
eegfile = os.path.join(
    data_path, example_sub, f'ses-{example_ses}', 'eeg',
    f'{example_sub}_ses-{example_ses}_bld{example_bld}_eeg_Bergen_CWreg_filt_ICA_rej.set'
)


from utils import plot_31ch_topomap

# Load saved SHAP results
D = np.load(out_file, allow_pickle=True)


# Find alpha band index
alpha_idx = D["band_names"].tolist().index("alpha")

# === Unsigned alpha (mean |SHAP|) ===
alpha_abs = D["imp_abs_cbb"][:, alpha_idx, :]      # shape: (channels, timebins)
vals31_alpha_abs = alpha_abs.sum(axis=1)           # sum over time → (channels,)

# === Signed alpha (mean signed SHAP) ===
alpha_signed = D["imp_signed_cbb"][:, alpha_idx, :]  # (channels, timebins)
vals31_alpha_signed = alpha_signed.sum(axis=1)       # sum over time

# Plot using your topomap function
plot_31ch_topomap(vals31_alpha_abs, eegfile, title="Alpha band — unsigned SHAP")
plot_31ch_topomap(vals31_alpha_signed, eegfile, title="Alpha band — signed SHAP")








D = np.load(out_file, allow_pickle=True)
vals31_signed = D["imp_signed_per_channel"]  # (31,)

# Optional: enforce symmetric limits for a diverging look
# (If your plot function returns the fig, you can set clim afterward.)
fig = plot_31ch_topomap(vals31_signed, eegfile, title="Mean signed SHAP per channel")
# Example (only if you tweak your plotter to return the image handle):
# vmax = np.max(np.abs(vals31_signed))
# im.set_clim(-vmax, vmax)



D = np.load(out_file, allow_pickle=True)
vals31 = D["imp_per_channel"]  # (31,)
plot_31ch_topomap(vals31, eegfile, title="Mean |SHAP| per channel (sum over band×time)")



b_alpha = D["band_names"].tolist().index("alpha")
vals31_alpha = D["imp_cbb"][:, b_alpha, :].sum(axis=1)
plot_31ch_topomap(vals31_alpha, eegfile, title="Mean |SHAP| per channel (sum over band×time)")


b_gamma = D["band_names"].tolist().index("gamma")
vals31_gamma = D["imp_cbb"][:, b_gamma, :].sum(axis=1)
plot_31ch_topomap(vals31_gamma, eegfile, title="Mean |SHAP| per channel (sum over band×time)")



