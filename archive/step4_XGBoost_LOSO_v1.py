import os
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# =========================
# Paths & config
# =========================
bands_file = os.path.join(
    os.getcwd(), "eegfmri_data_07122025", "processed_data", "eegfmri_data_bands_t2s.npz"
)
target_key = "Y_DAN"        # "Y_DMN", "Y_DNa", "Y_DNb"
random_state = 1337

# XGBoost config (shallow trees + regularization)
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
X_band = Z["X_band"]                 # (n_samples, n_channels, n_bands, n_bins) — already log-space
band_names = list(Z["band_names"])
bin_times = Z["bin_times"]
subject_ids = Z["subject_ids"]       # strings like 'sub-001'
Y_all = Z[target_key][:, 0]

n_samples, n_channels, n_bands, n_bins = X_band.shape
n_features = n_channels * n_bands * n_bins

print(f"Loaded X_band: {X_band.shape}, {len(np.unique(subject_ids))} subjects")


# =========================
# Subject-wise normalization (no log here)
# =========================
X_norm = np.zeros_like(X_band)
uniq_subs = np.unique(subject_ids)
for subj in uniq_subs:
    m = (subject_ids == subj)
    Xs = X_band[m]                       # (n_trials_subj, ch, bands, bins)
    Xs2d = Xs.reshape(Xs.shape[0], -1)
    scaler_subj = StandardScaler()
    Xs2d_scaled = scaler_subj.fit_transform(Xs2d)
    X_norm[m] = Xs2d_scaled.reshape(Xs.shape)

# Flatten for modeling
X_flat = X_norm.reshape(n_samples, -1)

# Helpers to map features back
def index_to_cbb(idx):
    ch, rem = divmod(idx, n_bands * n_bins)
    b, t = divmod(rem, n_bins)
    return ch, b, t

def feature_name(idx):
    ch, b, t = index_to_cbb(idx)
    return f"ch{ch:02d}|{band_names[b]}|t={bin_times[t]:.1f}s"

# =========================
# LOSO with per-fold scaling
# =========================
logo = LeaveOneGroupOut()
y_true_all, y_pred_all, fold_r2 = [], [], []

for fold, (tr, te) in enumerate(logo.split(X_flat, Y_all, groups=subject_ids), start=1):
    X_tr, X_te = X_flat[tr], X_flat[te]
    y_tr, y_te = Y_all[tr], Y_all[te]

    # Train-only scaling (keeps features comparable across folds)
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

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)
overall_r2 = r2_score(y_true_all, y_pred_all)
print(f"Overall LOSO R^2: {overall_r2:.3f}")

# =========================
# Feature importance (fit on ALL normalized data)
# =========================
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_flat)

model_all = XGBRegressor(**xgb_params)
model_all.fit(X_all_scaled, Y_all)

gain = model_all.get_booster().get_score(importance_type="gain")
imp = np.zeros(n_features, dtype=float)
for k, v in gain.items():  # keys like 'f0','f1',...
    idx = int(k[1:])
    if idx < n_features:
        imp[idx] = v

top_k = 25
top_idx = np.argsort(imp)[::-1][:top_k]
print("\nTop features by XGBoost gain:")
for rank, idx in enumerate(top_idx, start=1):
    print(f"{rank:2d}. {feature_name(idx)}   gain={imp[idx]:.4g}")

# Optional summaries
imp_cbb = imp.reshape(n_channels, n_bands, n_bins)
band_time_importance = np.sum(imp_cbb, axis=0)
chan_time_importance = np.sum(imp_cbb, axis=1)

print("\nBand x Time (sum gain over channels):")
for b, bname in enumerate(band_names):
    row = ", ".join([f"{bin_times[t]:.1f}s:{band_time_importance[b, t]:.3g}" for t in range(n_bins)])
    print(f"{bname:>6}: {row}")
