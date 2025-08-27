import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap  # pip install shap

# =========================
# Paths & config
# =========================
data_root = os.path.join(os.getcwd(), "eegfmri_data_07122025")
bands_file = os.path.join(
    data_root, "processed_data",
    "eeg_fmri_data_binned_2s_0to20s_canonicalbands.npz"
)
results_dir = os.path.join(os.getcwd(), "results")
os.makedirs(results_dir, exist_ok=True)

# Targets to run
TARGET_KEYS = ["Y_DAN", "Y_DMN", "Y_DNa", "Y_DNb"]
random_state = 1337

# OPTIONAL: permutation-null for |SHAP| (0 = skip)
N_PERM_NULL = 0         # e.g., 100–500 for paper; 0 for speed
PERM_SEED   = 1337

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
# Load data
# =========================
Z = np.load(bands_file, allow_pickle=True)

X_band      = Z["X_binned"]                 # (n_samples, n_channels, n_bands, n_bins)
band_names  = list(Z["bands"])
subject_ids = Z["subject_ids"]

# Time metadata (centers are negative)
if "time_bin_edges_sec" in Z.files:
    edges          = Z["time_bin_edges_sec"].astype(float)   # (n_bins, 2) [(0,2),(2,4),...,(18,20)]
    bin_times      = -edges.mean(axis=1)                     # [-1,-3,...,-19]
    bin_left_edges = edges[:, 0].astype(int)                 # [0,2,...,18]
    bin_right_edge = int(edges[-1, 1])                       # 20
else:
    n_bins_fallback = X_band.shape[-1]
    bin_times      = -(np.arange(n_bins_fallback) * 2 + 1).astype(float)
    bin_left_edges = np.arange(n_bins_fallback) * 2
    bin_right_edge = int(bin_left_edges[-1] + 2)

n_samples, n_channels, n_bands, n_bins = X_band.shape
unique_subs = np.unique(subject_ids)
print(f"Loaded X_binned: {X_band.shape} — subjects: {len(unique_subs)}")

# =========================
# Per-subject normalization (as before)
# =========================
X_norm = np.zeros_like(X_band)
for subj in unique_subs:
    m = (subject_ids == subj)
    Xs2d = X_band[m].reshape(np.sum(m), -1)
    scaler_subj = StandardScaler()
    X_norm[m] = scaler_subj.fit_transform(Xs2d).reshape(np.sum(m), n_channels, n_bands, n_bins)

# Flatten for XGB
X_flat = X_norm.reshape(n_samples, -1)

# =========================
# Helpers
# =========================
def make_explainer(model, X):
    """Robust TreeSHAP explainer."""
    try:
        expl = shap.Explainer(model)
        sv = expl(X)
        vals = sv.values
    except Exception:
        expl = shap.TreeExplainer(model)
        vals = expl.shap_values(X)
    return np.asarray(vals)

def subject_level_bt_shap(shap_values, subject_ids, unique_subs, n_channels, n_bands, n_bins):
    """
    SHAP -> subject × band × time by (mean over channels) then (mean over subject's samples).
    Returns signed and absolute arrays of shape (n_subj, n_bands, n_bins).
    """
    sv_cbt = shap_values.reshape(shap_values.shape[0], n_channels, n_bands, n_bins)
    n_subj = len(unique_subs)
    signed = np.zeros((n_subj, n_bands, n_bins), dtype=np.float32)
    absval = np.zeros_like(signed)
    for si, subj in enumerate(unique_subs):
        m = (subject_ids == subj)
        if not np.any(m):
            continue
        # mean over channels (axis=1) → (samples_subj, bands, bins), then mean over samples
        signed[si] = sv_cbt[m].mean(axis=1).mean(axis=0)
        absval[si] = np.abs(sv_cbt[m]).mean(axis=1).mean(axis=0)
    return signed, absval

def permute_within_subjects(y, subject_ids, rng):
    y_perm = y.copy()
    for subj in np.unique(subject_ids):
        m = (subject_ids == subj)
        y_perm[m] = rng.permutation(y_perm[m])
    return y_perm

# =========================
# Main loop (no folds; fit on ALL data)
# =========================
rng = np.random.default_rng(PERM_SEED)

for target_key in TARGET_KEYS:
    print("\n" + "="*70)
    print(f"Target (all-data SHAP): {target_key}")
    print("="*70)

    Y_all = Z[target_key][:, 0]  # (n_samples,)

    # Scale features once for the all-data model
    scaler_all = StandardScaler()
    X_all_scaled = scaler_all.fit_transform(X_flat)

    # Fit single model on all samples
    model_all = XGBRegressor(**xgb_params)
    model_all.fit(X_all_scaled, Y_all)

    # SHAP on all samples
    shap_values = make_explainer(model_all, X_all_scaled)        # (n_samples, n_features)

    # Global mean SHAP (across samples)
    shap_mean_abs    = np.abs(shap_values).mean(axis=0)          # (n_features,)
    shap_mean_signed = shap_values.mean(axis=0)                  # (n_features,)

    # Reshape to (channels, bands, time)
    imp_abs_cbb    = shap_mean_abs.reshape(n_channels, n_bands, n_bins)
    imp_signed_cbb = shap_mean_signed.reshape(n_channels, n_bands, n_bins)

    # Channel/band/time summaries
    imp_abs_per_channel    = imp_abs_cbb.sum(axis=(1, 2))
    imp_signed_per_channel = imp_signed_cbb.sum(axis=(1, 2))
    band_time_abs          = imp_abs_cbb.sum(axis=0)             # (n_bands, n_bins)
    band_time_signed       = imp_signed_cbb.sum(axis=0)
    chan_time_abs          = imp_abs_cbb.sum(axis=1)             # (n_channels, n_bins)
    chan_time_signed       = imp_signed_cbb.sum(axis=1)

    # Subject-level SHAP (subjects × bands × time)
    shap_signed_bt_by_subj, shap_abs_bt_by_subj = subject_level_bt_shap(
        shap_values, subject_ids, unique_subs, n_channels, n_bands, n_bins
    )

    # OPTIONAL: permutation-null for |SHAP| (empirical p-values over mean across subjects)
    p_abs_perm = np.array([])
    null_means_abs_bt = np.array([])
    if N_PERM_NULL > 0:
        print(f"Permutation null for |SHAP|: {N_PERM_NULL} permutations")
        null_means_abs_bt = np.zeros((N_PERM_NULL, n_bands, n_bins), dtype=np.float32)
        for r in range(N_PERM_NULL):
            y_perm = permute_within_subjects(Y_all, subject_ids, rng)
            model_null = XGBRegressor(**xgb_params)
            model_null.fit(X_all_scaled, y_perm)
            shap_null = make_explainer(model_null, X_all_scaled)
            _, abs_bt_by_subj_null = subject_level_bt_shap(
                shap_null, subject_ids, unique_subs, n_channels, n_bands, n_bins
            )
            null_means_abs_bt[r] = abs_bt_by_subj_null.mean(axis=0)  # (bands, bins)
            if (r + 1) % max(1, N_PERM_NULL // 10) == 0:
                print(f"  perm {r+1}/{N_PERM_NULL}")

        obs_mean_abs_bt = shap_abs_bt_by_subj.mean(axis=0)           # (bands, bins)
        ge_counts = (null_means_abs_bt >= obs_mean_abs_bt[None, ...]).sum(axis=0)
        p_abs_perm = (1.0 + ge_counts) / (N_PERM_NULL + 1.0)         # (bands, bins)
        print("Permutation p-values computed.")

    # ===== Save (keeps keys your plotting code expects). No folds/LOSO here. =====
    out_file = os.path.join(results_dir, f"xgb_shap_alldata_{target_key}_0to20s.npz")
    np.savez_compressed(
        out_file,
        # SHAP summaries (global)
        shap_mean_abs=shap_mean_abs,
        shap_mean_signed=shap_mean_signed,
        imp_abs_cbb=imp_abs_cbb,
        imp_signed_cbb=imp_signed_cbb,
        imp_abs_per_channel=imp_abs_per_channel,
        imp_signed_per_channel=imp_signed_per_channel,
        band_time_abs=band_time_abs,
        band_time_signed=band_time_signed,
        chan_time_abs=chan_time_abs,
        chan_time_signed=chan_time_signed,
        # Subject-level SHAP (bands × time)
        shap_signed_bt_by_subj=shap_signed_bt_by_subj,   # (n_subj, n_bands, n_bins)
        shap_abs_bt_by_subj=shap_abs_bt_by_subj,         # (n_subj, n_bands, n_bins)
        # Optional permutation-null summary for |SHAP|
        p_abs_perm=p_abs_perm,
        null_means_abs_bt=null_means_abs_bt,
        # Meta
        band_names=np.array(band_names),
        bin_times=np.array(bin_times, dtype=float),
        bin_left_edges=np.array(bin_left_edges, dtype=int),
        bin_right_edge=np.int64(bin_right_edge),
        subject_ids=subject_ids,
        unique_subjects=unique_subs,
        n_channels=n_channels, n_bands=n_bands, n_bins=n_bins,
        target_key=target_key,
        xgb_params=json.dumps(xgb_params),
        n_perm_null=int(N_PERM_NULL),
        perm_seed=int(PERM_SEED),
        # Back-compat placeholders (so other scripts don't break if they expect them)
        fold_r2=np.array([]), overall_r2=np.nan, fold_subjects=np.array([])
    )
    print(f"Saved: {out_file}")

print("\nDone. One model per target, trained on ALL data, SHAP saved.")
