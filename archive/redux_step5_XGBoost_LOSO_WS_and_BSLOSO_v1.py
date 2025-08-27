import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# =========================
# Paths & config
# =========================
data_root = os.path.join(os.getcwd(), "eegfmri_data_07122025")
in_file = os.path.join(
    data_root, "processed_data",
    "eeg_fmri_data_binned_2s_0to10s_canonicalbands.npz"
)
results_dir = os.path.join(os.getcwd(), "results")
os.makedirs(results_dir, exist_ok=True)

target_key = "Y_DNb"   # or "Y_DMN", "Y_DNa", "Y_DNb"
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
Z = np.load(in_file, allow_pickle=True)
X4 = Z["X_binned"]                         # (n_samples, n_channels, n_bands, n_bins)
Y   = Z[target_key][:, 0]
subs = Z["subject_ids"]                    # e.g., 'sub-001'
assert "session_ids" in Z.files, "Need session_ids saved in your NPZ."
sess = Z["session_ids"]                    # e.g., '001','002'

n_samples, n_channels, n_bands, n_bins = X4.shape
X = X4.reshape(n_samples, -1)              # flatten features

subjects = np.unique(subs)
print(f"Subjects: {subjects.tolist()}")

# =========================
# Helper: single paired fit/eval
# =========================
def paired_ws_bs_r2(subject_s, session_s, subject_u):
    """
    Train on ALL data except:
      - the held-out (subject_s, session_s)
      - ALL data from subject_u
    Then test:
      - WS on (subject_s, session_s)
      - BS on subject_u

    Returns dict with R2 and sizes.
    """
    if subject_s == subject_u:
        raise ValueError("subject_u must be different from subject_s.")

    te_ws_mask = (subs == subject_s) & (sess == session_s)
    te_bs_mask = (subs == subject_u)
    tr_mask    = ~(te_ws_mask | te_bs_mask)

    # Train-only scaling
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X[tr_mask])
    X_ws  = scaler.transform(X[te_ws_mask])
    X_bs  = scaler.transform(X[te_bs_mask])

    model = XGBRegressor(**xgb_params)
    model.fit(X_trs, Y[tr_mask])

    yhat_ws = model.predict(X_ws)
    yhat_bs = model.predict(X_bs)

    r2_ws = r2_score(Y[te_ws_mask], yhat_ws) if X_ws.shape[0] > 0 else np.nan
    r2_bs = r2_score(Y[te_bs_mask], yhat_bs) if X_bs.shape[0] > 0 else np.nan

    return dict(
        subject_ws=subject_s, session_ws=session_s,
        subject_bs=subject_u,
        n_train=int(tr_mask.sum()),
        n_ws=int(te_ws_mask.sum()), n_bs=int(te_bs_mask.sum()),
        r2_ws=float(r2_ws), r2_bs=float(r2_bs)
    )

# =========================
# Run all paired comparisons
# =========================
records = []
for s in subjects:
    s_sessions = np.unique(sess[subs == s])
    # skip subjects with <1 session held-out candidate
    for ses in s_sessions:
        for u in subjects:
            if u == s:
                continue
            rec = paired_ws_bs_r2(s, ses, u)
            records.append(rec)
            print(f"Train excl: ({s}, ses-{ses}) & {u}  ->  "
                  f"WS R2={rec['r2_ws']:.3f} (n={rec['n_ws']}), "
                  f"BS R2={rec['r2_bs']:.3f} (n={rec['n_bs']})")

# =========================
# Aggregate & save (robust to object dtype)
# =========================
# Build list of (subject, session) keys and get uniques via set of tuples
pairs = [(r["subject_ws"], r["session_ws"]) for r in records]
uniq_keys = sorted(set(pairs))  # list of unique (subject, session) tuples

avg_ws, avg_bs = [], []
for s, ses_i in uniq_keys:
    idx = [i for i, (ss, se) in enumerate(pairs) if (ss == s and se == ses_i)]
    ws_vals = np.array([records[i]["r2_ws"] for i in idx], dtype=float)
    bs_vals = np.array([records[i]["r2_bs"] for i in idx], dtype=float)
    avg_ws.append(np.nanmean(ws_vals))
    avg_bs.append(np.nanmean(bs_vals))

paired_diff = np.array([records[i]["r2_ws"] - records[i]["r2_bs"] for i in range(len(records))], dtype=float)

# Save everything
out_path = os.path.join(results_dir, f"xgb_ws_vs_bs_paired_{target_key}.npz")
np.savez_compressed(
    out_path,
    # raw per-pair results (aligned arrays)
    subject_ws=np.array([r["subject_ws"] for r in records], dtype=object),
    session_ws=np.array([r["session_ws"] for r in records], dtype=object),
    subject_bs=np.array([r["subject_bs"] for r in records], dtype=object),
    n_train=np.array([r["n_train"] for r in records], dtype=int),
    n_ws=np.array([r["n_ws"] for r in records], dtype=int),
    n_bs=np.array([r["n_bs"] for r in records], dtype=int),
    r2_ws=np.array([r["r2_ws"] for r in records], dtype=float),
    r2_bs=np.array([r["r2_bs"] for r in records], dtype=float),
    r2_ws_minus_bs=paired_diff,
    # per-(subject, session) averages over all u  (store as separate arrays)
    ws_avg_subject=np.array([s for s, _ in uniq_keys], dtype=object),
    ws_avg_session=np.array([se for _, se in uniq_keys], dtype=object),
    ws_avg_r2=np.array(avg_ws, dtype=float),
    bs_avg_r2=np.array(avg_bs, dtype=float),
    # meta
    n_channels=n_channels, n_bands=n_bands, n_bins=n_bins
)
print(f"\nSaved paired results to: {out_path}")

# Quick summary
print(f"\nOverall mean of paired (WS R2 - BS R2): {np.nanmean(paired_diff):.3f}")


