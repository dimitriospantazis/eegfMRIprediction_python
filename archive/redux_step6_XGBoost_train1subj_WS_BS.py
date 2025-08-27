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

target_key = "Y_DAN"   # or "Y_DMN", "Y_DNa", "Y_DNb"
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
# Load data
# =========================
Z = np.load(in_file, allow_pickle=True)

X4 = Z["X_binned"]                      # (n_samples, n_channels, n_bands, n_bins)  (already log-agg)
Y  = Z[target_key][:, 0]                # (n_samples,)
subs = Z["subject_ids"]                 # e.g., 'sub-001'
assert "session_ids" in Z.files, "session_ids not found in NPZ."
sess = Z["session_ids"]                 # e.g., '001','002'

n_samples, n_channels, n_bands, n_bins = X4.shape
X = X4.reshape(n_samples, -1)

subjects = np.unique(subs)
print(f"Subjects: {subjects.tolist()}")

def train_and_eval(X_tr, y_tr, X_te, y_te):
    """Train with train-only scaling; return predictions and R^2."""
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    model = XGBRegressor(**xgb_params)
    model.fit(X_trs, y_tr)
    y_hat = model.predict(X_tes)
    return y_hat, r2_score(y_te, y_hat), scaler, model

# =========================
# Loop: train on ONE subject, test WS and BS
# =========================
records = []  # one row per (train_subject s, ws_session ses, bs_subject u)
summary   = []  # one row per (s, ses) with WS R2 and pooled BS R2

for s in subjects:
    s_sessions = np.unique(sess[subs == s])
    if len(s_sessions) < 2:
        print(f"Skipping {s}: needs ≥2 sessions for WS.")
        continue

    for ses in s_sessions:
        # Train on subject s, sessions != ses
        tr_mask    = (subs == s) & (sess != ses)
        te_ws_mask = (subs == s) & (sess == ses)

        if tr_mask.sum() == 0 or te_ws_mask.sum() == 0:
            print(f"Skipping {s} ses-{ses}: no train/test samples.")
            continue

        # Fit once on this single-subject training set
        yhat_ws, r2_ws, scaler, model = train_and_eval(
            X[tr_mask], Y[tr_mask], X[te_ws_mask], Y[te_ws_mask]
        )
        print(f"[Train {s} (excluding ses-{ses})]  WS on ({s}, ses-{ses}): R^2={r2_ws:.3f}  n={te_ws_mask.sum()}")

        # Between-subject tests: each other subject u, same model/scaler
        bs_subjects = [u for u in subjects if u != s]
        bs_r2s = []
        bs_ns  = []
        for u in bs_subjects:
            te_bs_mask = (subs == u)
            X_bs = scaler.transform(X[te_bs_mask])
            yhat_bs = model.predict(X_bs)
            r2_bs = r2_score(Y[te_bs_mask], yhat_bs) if te_bs_mask.sum() else np.nan
            bs_r2s.append(r2_bs); bs_ns.append(int(te_bs_mask.sum()))
            records.append(dict(
                train_subject=s, ws_session=ses, bs_subject=u,
                n_train=int(tr_mask.sum()),
                n_ws=int(te_ws_mask.sum()), n_bs=int(te_bs_mask.sum()),
                r2_ws=float(r2_ws), r2_bs=float(r2_bs)
            ))
            print(f"    BS on {u}: R^2={r2_bs:.3f}  n={te_bs_mask.sum()}")

        # Pooled BS over all u != s
        te_bs_all = (subs != s)
        X_bs_all  = scaler.transform(X[te_bs_all])
        yhat_bs_all = model.predict(X_bs_all)
        r2_bs_pooled = r2_score(Y[te_bs_all], yhat_bs_all)

        summary.append(dict(
            train_subject=s, ws_session=ses,
            n_train=int(tr_mask.sum()),
            r2_ws=float(r2_ws),
            r2_bs_pooled=float(r2_bs_pooled),
            bs_subjects=np.array(bs_subjects, dtype=object),
            bs_r2=np.array(bs_r2s, dtype=float),
            bs_n =np.array(bs_ns, dtype=int)
        ))
        print(f"    BS pooled (all other subjects): R^2={r2_bs_pooled:.3f}  n={te_bs_all.sum()}")

# =========================
# Save
# =========================
out_path = os.path.join(results_dir, f"xgb_single_subject_train_WS_BS_{target_key}.npz")

# Flatten variable-length lists for saving
def stack_field(lst, key, dtype=object):
    return np.array([row[key] for row in lst], dtype=dtype)

np.savez_compressed(
    out_path,
    # per (s, ses, u)
    rec_train_subject = stack_field(records, "train_subject"),
    rec_ws_session    = stack_field(records, "ws_session"),
    rec_bs_subject    = stack_field(records, "bs_subject"),
    rec_n_train       = stack_field(records, "n_train", dtype=int),
    rec_n_ws          = stack_field(records, "n_ws", dtype=int),
    rec_n_bs          = stack_field(records, "n_bs", dtype=int),
    rec_r2_ws         = stack_field(records, "r2_ws", dtype=float),
    rec_r2_bs         = stack_field(records, "r2_bs", dtype=float),

    # per (s, ses) summary
    sum_train_subject = stack_field(summary, "train_subject"),
    sum_ws_session    = stack_field(summary, "ws_session"),
    sum_n_train       = stack_field(summary, "n_train", dtype=int),
    sum_r2_ws         = stack_field(summary, "r2_ws", dtype=float),
    sum_r2_bs_pooled  = stack_field(summary, "r2_bs_pooled", dtype=float),
    sum_bs_subjects   = stack_field(summary, "bs_subjects"),
    sum_bs_r2         = stack_field(summary, "bs_r2"),
    sum_bs_n          = stack_field(summary, "bs_n"),

    # meta
    target_key=target_key,
    n_channels=n_channels, n_bands=n_bands, n_bins=n_bins
)
print(f"\nSaved to: {out_path}")




























import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= Paths =========
data_root  = os.path.join(os.getcwd(), "eegfmri_data_07122025")
results_dir = os.path.join(os.getcwd(), "results")
target_key = "Y_DAN"  # change if needed
in_file = os.path.join(results_dir, f"xgb_single_subject_train_WS_BS_{target_key}.npz")

# ========= Load =========
D = np.load(in_file, allow_pickle=True)

subs   = D["sum_train_subject"]      # per (subject, held-out session)
sess   = D["sum_ws_session"]
ws_r2  = D["sum_r2_ws"].astype(float)
bs_r2  = D["sum_r2_bs_pooled"].astype(float)
n_tr   = D["sum_n_train"].astype(int)

assert subs.shape == sess.shape == ws_r2.shape == bs_r2.shape

# ========= Table =========
df = pd.DataFrame({
    "train_subject": subs,
    "ws_session": sess,
    "r2_ws": ws_r2,
    "r2_bs_pooled": bs_r2,
    "r2_diff_ws_minus_bs": ws_r2 - bs_r2,
    "n_train": n_tr
})
csv_path = os.path.join(results_dir, f"ws_bs_summary_{target_key}.csv")
df.to_csv(csv_path, index=False)
print(f"Saved summary CSV: {csv_path}")

# ========= Plots =========
# Common y-limits
ymin = np.nanmin([ws_r2.min(), bs_r2.min()])
ymax = np.nanmax([ws_r2.max(), bs_r2.max()])
pad  = 0.05 * (ymax - ymin + 1e-9)
ylim = (ymin - pad, ymax + pad)

# 1) Scatter: WS vs BS pooled (one point per (subject, session))
fig1, ax1 = plt.subplots(figsize=(5.2, 5.2))
ax1.scatter(bs_r2, ws_r2, s=40)
low = min(ylim[0], np.nanmin(bs_r2))
high = max(ylim[1], np.nanmax(bs_r2))
ax1.plot([low, high], [low, high], linestyle="--", linewidth=1)
ax1.set_xlabel("Between-subject R² (pooled)")
ax1.set_ylabel("Within-subject R²")
ax1.set_title(f"WS vs BS (trained on single subject) — {target_key}")
ax1.set_xlim(low, high); ax1.set_ylim(low, high)
fig1.tight_layout()
fig1.savefig(os.path.join(results_dir, f"ws_vs_bs_scatter_{target_key}.png"), dpi=150)

# 2) Paired lines per (subject, session)
labels = [f"{s}/ses-{se}" for s, se in zip(subs, sess)]
x = np.arange(len(labels))
fig2, ax2 = plt.subplots(figsize=(max(6, len(labels)*0.4), 4.2))
ax2.plot(x, ws_r2, marker="o", linestyle="", label="WS")
ax2.plot(x, bs_r2, marker="s", linestyle="", label="BS pooled")
for i in range(len(x)):
    ax2.plot([x[i], x[i]], [bs_r2[i], ws_r2[i]], color="gray", linewidth=0.8, alpha=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=60, ha="right")
ax2.set_ylabel("R²"); ax2.set_ylim(ylim)
ax2.set_title(f"Paired comparison per (subject,session) — {target_key}")
ax2.legend()
fig2.tight_layout()
fig2.savefig(os.path.join(results_dir, f"ws_bs_paired_{target_key}.png"), dpi=150)

# 3) Box/violin-style summary (simple boxplot with matplotlib)
fig3, ax3 = plt.subplots(figsize=(4.6, 4.2))
ax3.boxplot([ws_r2, bs_r2], labels=["WS", "BS pooled"], showfliers=False)
ax3.set_ylabel("R²"); ax3.set_ylim(ylim)
ax3.set_title(f"WS vs BS pooled — distribution across (subject,session)")
fig3.tight_layout()
fig3.savefig(os.path.join(results_dir, f"ws_bs_box_{target_key}.png"), dpi=150)

plt.show()

# ========= Optional: subject-level averages across sessions =========
g = df.groupby("train_subject")[["r2_ws", "r2_bs_pooled"]].mean().reset_index()
fig4, ax4 = plt.subplots(figsize=(max(5, len(g)*0.5), 4.0))
xx = np.arange(len(g))
w = 0.35
ax4.bar(xx - w/2, g["r2_ws"], width=w, label="WS (avg over sessions)")
ax4.bar(xx + w/2, g["r2_bs_pooled"], width=w, label="BS pooled (avg)")
ax4.set_xticks(xx); ax4.set_xticklabels(g["train_subject"], rotation=45, ha="right")
ax4.set_ylabel("Mean R²"); ax4.set_ylim(ylim)
ax4.set_title(f"Subject-level means — {target_key}")
ax4.legend()
fig4.tight_layout()
fig4.savefig(os.path.join(results_dir, f"ws_bs_subject_means_{target_key}.png"), dpi=150)










