import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import r2_score

# =========================
# Config
# =========================
model_type = "ridge"            # "ridge" or "lasso"
alpha_grid = np.logspace(-3, 3, 13)   # inner-CV search grid
bands_file = os.path.join(os.getcwd(), "eegfmri_data_07122025", "processed_data", "eegfmri_data_bands.npz")
out_dir = os.path.join(os.getcwd(), "results")
os.makedirs(out_dir, exist_ok=True)

targets = {
    "DAN": "Y_DAN",
    "DMN": "Y_DMN",
    "DNa": "Y_DNa",
    "DNb": "Y_DNb",
}

# =========================
# Load band-averaged data
# =========================
Z = np.load(bands_file, allow_pickle=True)
X_band = Z["X_band"]                # (n_samples, n_channels, n_bands, n_bins)
band_names = list(Z["band_names"])
bin_times = Z["bin_times"]
subject_ids = Z["subject_ids"]

n_samples, n_channels, n_bands, n_bins = X_band.shape
print(f"Loaded X_band: {X_band.shape}, bands: {band_names}")

# Flatten features: (samples, channels*bands*bins)
X_flat = X_band.reshape(n_samples, -1)

# =========================
# Helper: fit/predict with LOSO + inner CV for alpha
# =========================
def fit_loso_with_inner_cv(X, y, subjects, model_type="ridge", alpha_grid=None):
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    coefs_all = []
    alphas_chosen = []

    for fold, (tr_idx, te_idx) in enumerate(logo.split(X, y, groups=subjects), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        subj_tr = subjects[tr_idx]

        # scale using train only
        scaler = StandardScaler()
        X_trs = scaler.fit_transform(X_tr)
        X_tes = scaler.transform(X_te)

        # inner grouped CV to pick alpha
        # use GroupKFold on training subjects
        uniq_train_subj = np.unique(subj_tr)
        n_splits = min(5, len(uniq_train_subj)) if len(uniq_train_subj) > 1 else 2
        gkf = GroupKFold(n_splits=n_splits)

        best_alpha, best_score = None, -np.inf
        for a in alpha_grid:
            cv_scores = []
            for it_tr, it_va in gkf.split(X_trs, y_tr, groups=subj_tr):
                if model_type == "ridge":
                    mdl = Ridge(alpha=a)
                else:
                    mdl = Lasso(alpha=a, max_iter=10000)
                mdl.fit(X_trs[it_tr], y_tr[it_tr])
                y_hat = mdl.predict(X_trs[it_va])
                cv_scores.append(r2_score(y_tr[it_va], y_hat))
            m = np.mean(cv_scores)
            if m > best_score:
                best_score, best_alpha = m, a

        alphas_chosen.append(best_alpha)

        # refit on all training with best alpha
        if model_type == "ridge":
            mdl = Ridge(alpha=best_alpha)
        else:
            mdl = Lasso(alpha=best_alpha, max_iter=10000)

        mdl.fit(X_trs, y_tr)
        y_hat = mdl.predict(X_tes)

        y_true_all.append(y_te)
        y_pred_all.append(y_hat)
        coefs_all.append(mdl.coef_)  # (n_features,)

        print(f"Fold {fold:02d}: best alpha={best_alpha:.4g}, R2(val-in): {best_score:.3f}")

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    R2 = r2_score(y_true_all, y_pred_all)
    coefs_mean = np.mean(np.stack(coefs_all, axis=0), axis=0)   # average across outer folds

    return R2, coefs_mean, np.array(alphas_chosen), (y_true_all, y_pred_all)

# =========================
# Run for each target, save results & plots
# =========================
for name, key in targets.items():
    y = Z[key][:, 0]  # (n_samples,)

    R2, coef_mean, alphas, (y_true, y_pred) = fit_loso_with_inner_cv(
        X_flat, y, subject_ids, model_type=model_type, alpha_grid=alpha_grid
    )
    print(f"[{name}] LOSO R^2 = {R2:.3f} | median alpha = {np.median(alphas):.4g}")

    # reshape coefficients back to (channels, bands, bins)
    coef_cbT = coef_mean.reshape(n_channels, n_bands, n_bins)

    # --------- Feature importance summaries ---------
    # 1) Sum |coef| over channels -> (bands, bins)
    imp_band_time = np.sum(np.abs(coef_cbT), axis=0)

    # 2) Sum |coef| over bands -> (channels, bins)
    imp_chan_time = np.sum(np.abs(coef_cbT), axis=1)

    # --------- Plots ---------
    # Band x Time
    plt.figure(figsize=(10, 4))
    sns.heatmap(
        imp_band_time,
        cmap="viridis",
        xticklabels=[f"{t:.1f}" for t in bin_times],
        yticklabels=band_names
    )
    plt.title(f"{name}: |coef| summed over channels (Band x Time)")
    plt.xlabel("Time to event (s)")
    plt.ylabel("Band")
    plt.tight_layout()
    f1 = os.path.join(out_dir, f"{name}_importance_band_time.png")
    plt.savefig(f1, dpi=300)
    plt.close()

    # Channel x Time (optional quick look)
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        imp_chan_time,
        cmap="magma",
        xticklabels=[f"{t:.1f}" for t in bin_times],
        yticklabels=False
    )
    plt.title(f"{name}: |coef| summed over bands (Channel x Time)")
    plt.xlabel("Time to event (s)")
    plt.ylabel("Channels")
    plt.tight_layout()
    f2 = os.path.join(out_dir, f"{name}_importance_channel_time.png")
    plt.savefig(f2, dpi=300)
    plt.close()

    # Predicted vs True scatter
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, lw=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{name}: LOSO predictions (R2={R2:.3f})")
    plt.tight_layout()
    f3 = os.path.join(out_dir, f"{name}_y_true_vs_pred.png")
    plt.savefig(f3, dpi=300)
    plt.close()

    # Save numpy results for later analyses
    np.savez_compressed(
        os.path.join(out_dir, f"{name}_ridge_lasso_results.npz"),
        R2=R2,
        alphas=alphas,
        coef_mean=coef_mean,
        coef_reshaped=coef_cbT,
        band_names=np.array(band_names),
        bin_times=bin_times,
        y_true=y_true,
        y_pred=y_pred
    )

    print(f"[{name}] saved: \n  {f1}\n  {f2}\n  {f3}\n  {os.path.join(out_dir, f'{name}_ridge_lasso_results.npz')}")
