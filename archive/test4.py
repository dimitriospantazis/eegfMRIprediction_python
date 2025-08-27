import os
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

# ===== Config (simple) =====
bands_file = os.path.join(os.getcwd(), "eegfmri_data_07122025", "processed_data", "eegfmri_data_bands.npz")
model_type = "ridge"   # "ridge" or "lasso"
alpha = 10.0           # fixed regularization strength

# ===== Load band-averaged data =====
Z = np.load(bands_file, allow_pickle=True)
X_band = Z["X_band"]                # (n_samples, n_channels, n_bands, n_bins)
band_names = list(Z["band_names"])
bin_times = Z["bin_times"]
subject_ids = Z["subject_ids"]
Y_DAN = Z["Y_DAN"][:, 0]
Y_DMN = Z["Y_DMN"][:, 0]
Y_DNa = Z["Y_DNa"][:, 0]
Y_DNb = Z["Y_DNb"][:, 0]

n_samples, n_channels, n_bands, n_bins = X_band.shape
X_flat = X_band.reshape(n_samples, -1)  # (samples, features)

def loso_linear(X, y, groups, model_type="ridge", alpha=10.0):
    """
    Leave-One-Subject-Out linear model with fixed alpha.
    Returns: overall_R2, per_fold_R2(list), mean_coef (reshaped to ch x band x bin)
    """
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    per_fold_r2 = []
    coefs = []

    fold = 0
    for tr, te in logo.split(X, y, groups=groups):
        fold += 1
        # Scale X using train only
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        y_tr, y_te = y[tr], y[te]

        # Choose model
        if model_type == "ridge":
            model = Ridge(alpha=alpha)
        elif model_type == "lasso":
            model = Lasso(alpha=alpha, max_iter=10000)
        else:
            raise ValueError("model_type must be 'ridge' or 'lasso'")

        # Fit + predict
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)

        # Metrics
        r2 = r2_score(y_te, y_hat)
        per_fold_r2.append(r2)
        print(f"Fold {fold:02d}  R2: {r2:.3f}")

        # Collect
        y_true_all.append(y_te)
        y_pred_all.append(y_hat)
        coefs.append(model.coef_)  # (n_features,)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    overall_R2 = r2_score(y_true_all, y_pred_all)
    coef_mean = np.mean(np.stack(coefs, axis=0), axis=0)  # average across folds

    # Reshape coefficients back to (channels, bands, bins)
    coef_reshaped = coef_mean.reshape(n_channels, n_bands, n_bins)

    print(f"Overall LOSO R2: {overall_R2:.3f}")
    return overall_R2, per_fold_r2, coef_reshaped

# ===== Run (example: DAN). Repeat for DMN/DNa/DNb as needed. =====
print("\n=== DAN ===")
R2_DAN, r2_folds_DAN, coef_DAN = loso_linear(X_flat, Y_DAN, subject_ids, model_type=model_type, alpha=alpha)

# (Optional) run others quickly:
print("\n=== DMN ===")
R2_DMN, r2_folds_DMN, coef_DMN = loso_linear(X_flat, Y_DMN, subject_ids, model_type=model_type, alpha=alpha)

print("\n=== DNa ===")
R2_DNa, r2_folds_DNa, coef_DNa = loso_linear(X_flat, Y_DNa, subject_ids, model_type=model_type, alpha=alpha)

print("\n=== DNb ===")
R2_DNb, r2_folds_DNb, coef_DNb = loso_linear(X_flat, Y_DNb, subject_ids, model_type=model_type, alpha=alpha)

# coef_* tensors are (n_channels, n_bands, n_bins)
# You can quickly summarize importance as:
# np.sum(np.abs(coef_DAN), axis=0) -> (bands x bins)
# np.sum(np.abs(coef_DAN), axis=1) -> (channels x bins)




