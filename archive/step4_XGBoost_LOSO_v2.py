import os
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import xgboost as xgb

# =========================
# Config
# =========================
bands_file = os.path.join(
    os.getcwd(), "eegfmri_data_07122025", "processed_data", "eegfmri_data_bands_t2s.npz"
)
target_key = "Y_DAN"   # or "Y_DMN", "Y_DNa", "Y_DNb"

# XGBoost params (stronger regularization for cross-subject generalization)
xgb_params = dict(
    max_depth=2,
    min_child_weight=10,
    reg_lambda=5.0,
    reg_alpha=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    learning_rate=0.03,
    n_estimators=2000,             # upper bound; early stopping will pick best_iteration_
    objective="reg:squarederror",
    tree_method="hist",
    random_state=1337,
    n_jobs=0
)

# =========================
# Helper: version-proof early stopping
# =========================
def fit_with_es(model, X_tr, y_tr, X_va, y_va, rounds=50):
    """Fit with early stopping; compatible across XGBoost versions."""
    try:
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            early_stopping_rounds=rounds,
            verbose=False
        )
        return model
    except TypeError:
        try:
            from xgboost.callback import EarlyStopping
            cb = EarlyStopping(
                rounds=rounds, save_best=True,
                maximize=False,          # rmse -> lower is better
                data_name="validation_0",
                metric_name="rmse"
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="rmse",
                callbacks=[cb],
                verbose=False
            )
            return model
        except Exception as e:
            print(f"[warn] Early stopping not available ({e}). Training without ES.")
            model.set_params(n_estimators=400)   # safer cap
            model.fit(X_tr, y_tr)
            return model

# =========================
# Load data
# =========================
Z = np.load(bands_file, allow_pickle=True)
X_band = Z["X_band"]                 # (n_samples, n_channels, n_bands, n_bins) — already log-space
band_names = list(Z["band_names"])
bin_times = Z["bin_times"]           # 2s-centered times, e.g., [-0.5, -2.5, ...]
subject_ids = Z["subject_ids"]       # strings like 'sub-001'
Y_all = Z[target_key][:, 0]

n_samples, n_channels, n_bands, n_bins = X_band.shape
n_features = n_channels * n_bands * n_bins
print(f"Data: {X_band.shape}, subjects={len(np.unique(subject_ids))}, features={n_features}")

# =========================
# Sanitize any non-finite values (rare)
# =========================
X_flat_tmp = X_band.reshape(n_samples, -1)
bad = ~np.isfinite(X_flat_tmp)
if bad.any():
    col_means = np.nanmean(np.where(np.isfinite(X_flat_tmp), X_flat_tmp, np.nan), axis=0)
    X_flat_tmp[bad] = np.take(col_means, np.where(bad)[1])
    X_band = X_flat_tmp.reshape(n_samples, n_channels, n_bands, n_bins)

# =========================
# Subject-wise z-scoring (no extra log)
# =========================
X_norm = np.zeros_like(X_band)
for subj in np.unique(subject_ids):
    m = (subject_ids == subj)
    Xs = X_band[m]                                  # (trials_subj, ch, bands, bins)
    Xs2d = Xs.reshape(Xs.shape[0], -1)
    scaler_subj = StandardScaler()
    Xs2d_scaled = scaler_subj.fit_transform(Xs2d)
    X_norm[m] = Xs2d_scaled.reshape(Xs.shape)

# Flatten normalized features
X_flat = X_norm.reshape(n_samples, -1)

# =========================
# LOSO CV with subject-based early stopping
# =========================
logo = LeaveOneGroupOut()
y_true_all, y_pred_all, fold_r2 = [], [], []

for fold, (tr, te) in enumerate(logo.split(X_flat, Y_all, groups=subject_ids), start=1):
    X_tr, X_te = X_flat[tr], X_flat[te]
    y_tr, y_te = Y_all[tr], Y_all[te]
    subj_tr = subject_ids[tr]

    # Train-only scaling
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    # Validation split that holds out entire subjects (mimics LOSO)
    uniq_train_subj = np.unique(subj_tr)
    n_splits = max(2, min(5, len(uniq_train_subj)))   # 2..5 depending on how many subjects in train
    gkf = GroupKFold(n_splits=n_splits)
    tr_in, tr_va = next(gkf.split(X_trs, y_tr, groups=subj_tr))  # first split is enough

    # Fit with early stopping
    model = XGBRegressor(**xgb_params)
    model = fit_with_es(model, X_trs[tr_in], y_tr[tr_in], X_trs[tr_va], y_tr[tr_va])

    # Predict & score
    y_hat = model.predict(X_tes)
    r2 = r2_score(y_te, y_hat)
    fold_r2.append(r2)
    y_true_all.append(y_te)
    y_pred_all.append(y_hat)
    best_iter = getattr(model, "best_iteration", None)
    print(f"Fold {fold:02d}  R2: {r2:.3f}  " + (f"(best_iter={best_iter})" if best_iter is not None else ""))

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)
overall_r2 = r2_score(y_true_all, y_pred_all)
print(f"\nOverall LOSO R^2: {overall_r2:.3f}")

# =========================
# Global feature importance (fit once on ALL normalized+scaled data)
# =========================
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_flat)

# Build global validation (subject-held-out split) for early stopping too
uniq_all_subj = np.unique(subject_ids)
n_splits_all = max(2, min(5, len(uniq_all_subj)))
gkf_all = GroupKFold(n_splits_all)
tr_in_all, tr_va_all = next(gkf_all.split(X_all_scaled, Y_all, groups=subject_ids))

model_all = XGBRegressor(**xgb_params)
model_all = fit_with_es(model_all, X_all_scaled[tr_in_all], Y_all[tr_in_all],
                        X_all_scaled[tr_va_all], Y_all[tr_va_all])

# Extract "gain" importance
gain = model_all.get_booster().get_score(importance_type="gain")
imp = np.zeros(n_features, dtype=float)
for k, v in gain.items():            # keys like 'f0','f1',...
    idx = int(k[1:])
    if idx < n_features:
        imp[idx] = v

# Map feature index -> (channel, band, time) pretty names
def index_to_cbb(idx):
    ch, rem = divmod(idx, n_bands * n_bins)
    b, t = divmod(rem, n_bins)
    return ch, b, t

def feature_name(idx):
    ch, b, t = index_to_cbb(idx)
    return f"ch{ch:02d}|{band_names[b]}|t={bin_times[t]:.1f}s"

# Print top features
top_k = 25
top_idx = np.argsort(imp)[::-1][:top_k]
print("\nTop features by XGBoost gain:")
for rank, idx in enumerate(top_idx, start=1):
    print(f"{rank:2d}. {feature_name(idx)}   gain={imp[idx]:.4g}")

# Optional summaries
imp_cbb = imp.reshape(n_channels, n_bands, n_bins)
band_time_importance = np.sum(imp_cbb, axis=0)   # (n_bands, n_bins)
chan_time_importance = np.sum(imp_cbb, axis=1)   # (n_channels, n_bins)

print("\nBand x Time (sum gain over channels):")
for b, bname in enumerate(band_names):
    row = ", ".join([f"{bin_times[t]:.1f}s:{band_time_importance[b, t]:.3g}" for t in range(n_bins)])
    print(f"{bname:>6}: {row}")
