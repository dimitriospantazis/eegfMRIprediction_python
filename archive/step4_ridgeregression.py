import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Load and Preprocess Data ----------
load_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025', 'processed_data', 'eeg_fmri_data.npz')
data = np.load(load_path, allow_pickle=True)

X = data['X']           # (n_samples, n_channels, n_freqs, n_bins)
Y_DAN = data['Y_DAN']   # (n_samples, 1)
Y_DMN = data['Y_DMN']
Y_DNa = data['Y_DNa']
Y_DNb = data['Y_DNb']
subject_ids = data['subject_ids']
n_samples, n_channels, n_freqs, n_bins = X.shape

epsilon = 1e-10
X_log = np.log10(X + epsilon)

# ---------- Frequency axis ----------
freqs = np.linspace(0, 40, n_freqs)

# ---------- Canonical bands ----------
band_defs = [
    ("Delta", 1.0, 4.0),
    ("Theta", 4.0, 8.0),
    ("Alpha", 8.0, 12.0),
    ("Beta", 13.0, 30.0),
    ("Gamma", 30.0, 40.1),
]
band_names = [b[0] for b in band_defs]
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
        band_feats.append(np.zeros((n_samples, n_channels, 1, n_bins)))
    else:
        band_feats.append(np.mean(X_log[:, :, idx, :], axis=2, keepdims=True))
X_band = np.concatenate(band_feats, axis=2)  # (n_samples, n_channels, n_bands, n_bins)
n_bands = X_band.shape[2]

# ---------- Flatten for regression ----------
# Shape: (n_samples, n_channels*n_bands*n_bins)
X_flat = X_band.reshape(n_samples, -1)

# ---------- Z-score features ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

# ---------- Predictive modeling function ----------
def ridge_predict_and_analyze(X_scaled, Y, subject_ids, model_type='ridge', alpha=1.0):
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    coefs_all = []

    for train_idx, test_idx in logo.split(X_scaled, Y, groups=subject_ids):
        if model_type == 'ridge':
            model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha, max_iter=5000)
        else:
            raise ValueError("model_type must be 'ridge' or 'lasso'")
        
        model.fit(X_scaled[train_idx], Y[train_idx])
        y_pred = model.predict(X_scaled[test_idx])

        y_true_all.append(Y[test_idx])
        y_pred_all.append(y_pred)
        coefs_all.append(model.coef_)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)
    coefs_mean = np.mean(np.stack(coefs_all, axis=0), axis=0)
    
    # Reshape back to (channels, bands, bins)
    coefs_reshaped = coefs_mean.reshape(n_channels, n_bands, n_bins)
    return r2, coefs_reshaped

# ---------- Run for DAN ----------
r2_DAN, coefs_DAN = ridge_predict_and_analyze(X_scaled, Y_DAN[:,0], subject_ids, model_type='ridge', alpha=10.0)
print(f"DAN R^2 (LOSO CV): {r2_DAN:.3f}")

# ---------- Visualize coefficients ----------
# Example: sum of absolute coefs over channels
coef_band_time = np.sum(np.abs(coefs_DAN), axis=0)  # (n_bands, n_bins)

plt.figure(figsize=(10,4))
sns.heatmap(coef_band_time, xticklabels=np.round(np.linspace(bin_times.min(), bin_times.max(), n_bins),1),
            yticklabels=band_names, cmap='viridis')
plt.xlabel('Time to event (s)')
plt.ylabel('Band')
plt.title('Feature importance (sum over channels)')
plt.show()




