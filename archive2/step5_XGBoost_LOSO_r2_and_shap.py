import os
import json
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
    "eeg_fmri_data_binned_2s_0to20s_canonicalbands.npz"
)
results_dir = os.path.join(os.getcwd(), "results")
os.makedirs(results_dir, exist_ok=True)

# Run all four targets
TARGET_KEYS = ["Y_DAN", "Y_DMN", "Y_DNa", "Y_DNb"]
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
# Load X and metadata (shared across targets)
# =========================
Z = np.load(bands_file, allow_pickle=True)

X_band = Z["X_binned"]                               # (n_samples, n_channels, n_bands, n_bins)
band_names = list(Z["bands"])                        # ['delta','theta','alpha','beta','gamma']
subject_ids = Z["subject_ids"]

# Bin meta (edges + centers; centers are negative lag for naming)
if "time_bin_edges_sec" in Z.files:
    edges = Z["time_bin_edges_sec"].astype(float)    # (10,2): [(0,2),(2,4),...,(18,20)]
    bin_times = -edges.mean(axis=1)                  # centers: [-1,-3,...,-19]
    bin_left_edges = edges[:, 0].astype(int)         # [0,2,...,18]
    bin_right_edge = int(edges[-1, 1])               # 20
else:
    n_bins_fallback = X_band.shape[-1]
    bin_times = -(np.arange(n_bins_fallback) * 2 + 1).astype(float)
    bin_left_edges = np.arange(n_bins_fallback) * 2
    bin_right_edge = int(bin_left_edges[-1] + 2)

n_samples, n_channels, n_bands, n_bins = X_band.shape
unique_subs = np.unique(subject_ids)
print(f"Loaded X_binned: {X_band.shape} — subjects: {len(unique_subs)}")

# =========================
# Subject-wise normalization (shared, independent of target)
# =========================
X_norm = np.zeros_like(X_band)
for subj in unique_subs:
    m = (subject_ids == subj)
    Xs2d = X_band[m].reshape(np.sum(m), -1)
    scaler_subj = StandardScaler()
    X_norm[m] = scaler_subj.fit_transform(Xs2d).reshape(np.sum(m), n_channels, n_bands, n_bins)

X_flat = X_norm.reshape(n_samples, -1)

# =========================
# Helpers for feature names (optional, kept from your version)
# =========================
def index_to_cbb(idx):
    ch, rem = divmod(idx, n_bands * n_bins)
    b, t = divmod(rem, n_bins)
    return ch, b, t

def feature_name(idx):
    ch, b, t = index_to_cbb(idx)
    return f"ch{ch:02d}|{band_names[b]}|t={bin_times[t]:.1f}s"

# =========================
# Loop over targets
# =========================
summary_rows = []
for target_key in TARGET_KEYS:
    print("\n" + "="*70)
    print(f"Target: {target_key}")
    print("="*70)

    Y_all = Z[target_key][:, 0]  # (n_samples,)

    # ========== LOSO (per-fold scaling) ==========
    logo = LeaveOneGroupOut()
    oof_pred = np.full_like(Y_all, fill_value=np.nan, dtype=float)
    oof_true = Y_all.copy()
    fold_r2 = []
    fold_subjects = []

    for fold, (tr, te) in enumerate(logo.split(X_flat, Y_all, groups=subject_ids), start=1):
        X_tr, X_te = X_flat[tr], X_flat[te]
        y_tr, y_te = Y_all[tr], Y_all[te]

        scaler_fold = StandardScaler()
        X_trs = scaler_fold.fit_transform(X_tr)
        X_tes = scaler_fold.transform(X_te)

        model = XGBRegressor(**xgb_params)
        model.fit(X_trs, y_tr)

        y_hat = model.predict(X_tes)
        oof_pred[te] = y_hat

        r2 = r2_score(y_te, y_hat)
        fold_r2.append(r2)
        fold_subjects.append(subject_ids[te][0])
        print(f"Fold {fold:02d}  subj={subject_ids[te][0]}  R^2: {r2:.3f}")

    overall_r2 = r2_score(oof_true, oof_pred)
    print(f"Overall LOSO R^2: {overall_r2:.3f}")

    # ========== Fit ALL-data model & SHAP ==========
    scaler_all = StandardScaler()
    X_all_scaled = scaler_all.fit_transform(X_flat)

    model_all = XGBRegressor(**xgb_params)
    model_all.fit(X_all_scaled, Y_all)

    # TreeSHAP
    try:
        explainer = shap.Explainer(model_all)   # TreeSHAP path
        sv = explainer(X_all_scaled)
        shap_values = sv.values
    except Exception:
        explainer = shap.TreeExplainer(model_all)
        shap_values = explainer.shap_values(X_all_scaled)

    shap_values = np.asarray(shap_values)  # (n_samples, n_features)

    # ---- Global (across all samples) means, like before ----
    shap_mean_abs    = np.abs(shap_values).mean(axis=0)   # (n_features,)
    shap_mean_signed = shap_values.mean(axis=0)           # (n_features,)

    # Reshape & aggregate to (channels, bands, time)
    imp_abs_cbb    = shap_mean_abs.reshape(n_channels, n_bands, n_bins)
    imp_signed_cbb = shap_mean_signed.reshape(n_channels, n_bands, n_bins)

    imp_abs_per_channel    = imp_abs_cbb.sum(axis=(1, 2))
    imp_signed_per_channel = imp_signed_cbb.sum(axis=(1, 2))
    band_time_abs          = imp_abs_cbb.sum(axis=0)      # (n_bands, n_bins)
    band_time_signed       = imp_signed_cbb.sum(axis=0)
    chan_time_abs          = imp_abs_cbb.sum(axis=1)      # (n_channels, n_bins)
    chan_time_signed       = imp_signed_cbb.sum(axis=1)

    # ---- NEW: Subject-level SHAP (bands × time) ----
    # reshape signed SHAP to (samples, channels, bands, time)
    shap_vals_cbt = shap_values.reshape(n_samples, n_channels, n_bands, n_bins)

    n_subj = len(unique_subs)
    shap_signed_bt_by_subj = np.zeros((n_subj, n_bands, n_bins), dtype=np.float32)
    shap_abs_bt_by_subj    = np.zeros((n_subj, n_bands, n_bins), dtype=np.float32)

    for si, subj in enumerate(unique_subs):
        m = (subject_ids == subj)
        if not np.any(m):
            continue
        # signed: mean over channels, then mean over the subject's samples
        subj_signed = shap_vals_cbt[m].mean(axis=1)         # (n_samples_subj, n_bands, n_bins)
        shap_signed_bt_by_subj[si] = subj_signed.mean(axis=0)

        # absolute: same but with abs
        subj_abs = np.abs(shap_vals_cbt[m]).mean(axis=1)    # (n_samples_subj, n_bands, n_bins)
        shap_abs_bt_by_subj[si] = subj_abs.mean(axis=0)

    # Save per-target NPZ
    out_file = os.path.join(results_dir, f"xgb_loso_r2_and_shap_{target_key}_0to20s.npz")
    np.savez_compressed(
        out_file,
        # OOF predictions and fold metrics
        oof_true=oof_true,
        oof_pred=oof_pred,
        subject_ids=subject_ids,
        unique_subjects=unique_subs,
        fold_r2=np.array(fold_r2, dtype=float),
        fold_subjects=np.array(fold_subjects),
        overall_r2=float(overall_r2),
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
        # NEW: subject-level SHAP (bands × time)
        shap_signed_bt_by_subj=shap_signed_bt_by_subj,   # (n_subj, n_bands, n_bins)
        shap_abs_bt_by_subj=shap_abs_bt_by_subj,         # (n_subj, n_bands, n_bins)
        # Meta
        band_names=np.array(band_names),
        bin_times=np.array(bin_times, dtype=float),           # centers (-1,-3,...,-19)
        bin_left_edges=np.array(bin_left_edges, dtype=int),   # [0,2,...,18]
        bin_right_edge=np.int64(bin_right_edge),
        n_channels=n_channels, n_bands=n_bands, n_bins=n_bins,
        target_key=target_key,
        xgb_params=json.dumps(xgb_params)
    )
    print(f"Saved: {out_file}")

    # Add to summary
    r2_arr = np.array(fold_r2, dtype=float)
    summary_rows.append(dict(
        target_key=target_key,
        n_subjects=len(unique_subs),
        mean_fold_r2=float(r2_arr.mean()),
        sd_fold_r2=float(r2_arr.std(ddof=1)),
        median_fold_r2=float(np.median(r2_arr)),
        overall_r2=float(overall_r2),
        positives=int((r2_arr > 0).sum())
    ))

# Save a small TSV summary
summary_path = os.path.join(results_dir, "loso_r2_summary_0to20s.tsv")
with open(summary_path, "w") as f:
    cols = ["target_key","n_subjects","mean_fold_r2","sd_fold_r2","median_fold_r2","overall_r2","positives"]
    f.write("\t".join(cols) + "\n")
    for row in summary_rows:
        f.write("\t".join(str(row[c]) for c in cols) + "\n")
print(f"Summary table: {summary_path}")
