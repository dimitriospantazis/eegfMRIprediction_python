"""
Compare Within-Subject (WS) vs Between-Subject (BS pooled) performance
for models trained on a single subject.

Inputs
------
Loads:  .../results/xgb_single_subject_train_WS_BS_<TARGET>.npz
(from the "single-subject train" script you ran earlier)

Outputs
-------
- CSV:   ws_bs_summary_<TARGET>.csv
- PNGs:  ws_vs_bs_scatter_<TARGET>.png
         ws_bs_paired_<TARGET>.png
         ws_bs_box_<TARGET>.png
         ws_bs_delta_hist_<TARGET>.png
- Console stats: mean ΔR², SE, paired t-test, Wilcoxon, sign test, bootstrap CI
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =============== Paths/config ===============
results_dir = os.path.join(os.getcwd(), "results")
os.makedirs(results_dir, exist_ok=True)

target_key = "Y_DAN"  # change if needed
in_file = os.path.join(results_dir, f"xgb_single_subject_train_WS_BS_{target_key}.npz")

# =============== Load ===============
D = np.load(in_file, allow_pickle=True)

subs  = D["sum_train_subject"].astype(str)       # per (subject, held-out session)
sess  = D["sum_ws_session"].astype(str)
ws_r2 = D["sum_r2_ws"].astype(float)
bs_r2 = D["sum_r2_bs_pooled"].astype(float)
n_tr  = D["sum_n_train"].astype(int)

assert subs.shape == sess.shape == ws_r2.shape == bs_r2.shape
labels = [f"{s}/ses-{se}" for s, se in zip(subs, sess)]

# =============== Table ===============
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

# =============== Paired stats on ΔR² ===============
delta = ws_r2 - bs_r2
n = np.sum(~np.isnan(delta))
mean_delta = np.nanmean(delta)
se_delta = np.nanstd(delta, ddof=1) / np.sqrt(n)
pos_frac = np.nanmean(delta > 0) * 100.0

# t-test and Wilcoxon (handle constant-zero edge cases)
try:
    p_t = stats.ttest_rel(ws_r2, bs_r2, nan_policy='omit').pvalue
except Exception:
    p_t = np.nan
try:
    # Wilcoxon requires non-all-zero diffs
    nonzero = delta[~np.isnan(delta)]
    if np.allclose(nonzero, 0):
        p_wilcoxon = np.nan
    else:
        p_wilcoxon = stats.wilcoxon(ws_r2, bs_r2, zero_method="wilcox", alternative="two-sided").pvalue
except Exception:
    p_wilcoxon = np.nan

# Sign test (binomial, two-sided)
k = int(np.sum(delta > 0))
p_sign = stats.binomtest(k=k, n=n, p=0.5, alternative="two-sided").pvalue if n > 0 else np.nan

# Bootstrap 95% CI for mean ΔR²
def bootstrap_mean_ci(x, B=10000, seed=0):
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)

ci_lo, ci_hi = bootstrap_mean_ci(delta)

print("\n===== Paired comparison: WS – BS =====")
print(f"ΔR² mean ± SE  : {mean_delta:.3f} ± {se_delta:.3f}")
print(f"Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"ΔR² > 0 folds  : {pos_frac:.1f}%")
print(f"Paired t-test p : {p_t:.4g}")
print(f"Wilcoxon p      : {p_wilcoxon:.4g}")
print(f"Sign test p     : {p_sign:.4g}")

# =============== Optional: association/accuracy metrics if present ===============
if "ws_y_true" in D.files and "ws_y_pred" in D.files:
    from sklearn.metrics import mean_squared_error
    y_true_ws = D["ws_y_true"]; y_pred_ws = D["ws_y_pred"]
    r_ws = np.corrcoef(y_true_ws, y_pred_ws)[0,1] if len(y_true_ws) > 2 else np.nan
    nrmse_ws = np.sqrt(mean_squared_error(y_true_ws, y_pred_ws)) / np.std(y_true_ws)
    print(f"\n[Optional WS metrics on pooled preds]  Pearson r={r_ws:.3f}, NRMSE={nrmse_ws:.3f}")

if "bs_y_true" in D.files and "bs_y_pred" in D.files:
    from sklearn.metrics import mean_squared_error
    y_true_bs = D["bs_y_true"]; y_pred_bs = D["bs_y_pred"]
    r_bs = np.corrcoef(y_true_bs, y_pred_bs)[0,1] if len(y_true_bs) > 2 else np.nan
    nrmse_bs = np.sqrt(mean_squared_error(y_true_bs, y_pred_bs)) / np.std(y_true_bs)
    print(f"[Optional BS metrics on pooled preds] Pearson r={r_bs:.3f}, NRMSE={nrmse_bs:.3f}")

# =============== Plots ===============
def compute_limits(a, b):
    amin = np.nanmin([a.min(), b.min()])
    amax = np.nanmax([a.max(), b.max()])
    pad = 0.05 * (amax - amin + 1e-9)
    low, high = amin - pad, amax + pad
    return low, high

# 1) Scatter WS vs BS with identity
fig1, ax1 = plt.subplots(figsize=(5.4, 5.4))
low, high = compute_limits(ws_r2, bs_r2)
ax1.scatter(bs_r2, ws_r2, s=36, alpha=0.9)
ax1.plot([low, high], [low, high], '--', lw=1)
ax1.set_xlabel("Between-subject R² (pooled)")
ax1.set_ylabel("Within-subject R²")
ax1.set_title(f"WS vs BS (single-subject trained) — {target_key}\nR²<0 = worse than mean baseline")
ax1.set_xlim(low, high); ax1.set_ylim(low, high)
fig1.tight_layout()
fig1.savefig(os.path.join(results_dir, f"ws_vs_bs_scatter_{target_key}.png"), dpi=150)

# 2) Paired lines per subject (session-averaged)
# Average WS and BS over sessions for each subject
g = df.groupby("train_subject", as_index=False).agg(
    ws_avg=("r2_ws", "mean"),
    bs_avg=("r2_bs_pooled", "mean"),
)
labels_subj = g["train_subject"].to_numpy()
ws_avg = g["ws_avg"].to_numpy()
bs_avg = g["bs_avg"].to_numpy()

x = np.arange(len(labels_subj))
fig2, ax2 = plt.subplots(figsize=(max(6, len(labels_subj)*0.45), 4.4))
ax2.plot(x, ws_avg, 'o', label="WS (avg over sessions)")
ax2.plot(x, bs_avg, 's', label="BS pooled (avg over sessions)")
for i in range(len(x)):
    ax2.plot([x[i], x[i]], [bs_avg[i], ws_avg[i]], color="gray", lw=0.8, alpha=0.8)

ax2.set_xticks(x)
ax2.set_xticklabels(labels_subj, rotation=45, ha="right")
ylow, yhigh = compute_limits(ws_avg, bs_avg)
ax2.set_ylim(ylow, yhigh)
ax2.set_ylabel("R²")
ax2.set_title(f"Paired comparison per subject (session-averaged) — {target_key}")
ax2.legend()
fig2.tight_layout()
fig2.savefig(os.path.join(results_dir, f"ws_bs_paired_subject_avg_{target_key}.png"), dpi=150)


# 3) Boxplot WS vs BS
fig3, ax3 = plt.subplots(figsize=(4.8, 4.4))
ax3.boxplot([ws_r2, bs_r2], labels=["WS", "BS pooled"], showfliers=False)
ylow, yhigh = compute_limits(ws_r2, bs_r2)
ax3.set_ylim(ylow, yhigh)
ax3.set_ylabel("R²")
ax3.set_title("WS vs BS pooled — distribution across (subject, session)")
fig3.tight_layout()
fig3.savefig(os.path.join(results_dir, f"ws_bs_box_{target_key}.png"), dpi=150)

# 4) Histogram of ΔR²
fig4, ax4 = plt.subplots(figsize=(5.6, 4.2))
ax4.hist(delta[~np.isnan(delta)], bins=15, edgecolor='black', alpha=0.8)
ax4.axvline(0, color='k', linestyle='--', lw=1)
ax4.set_xlabel("ΔR² = R²(WS) − R²(BS pooled)")
ax4.set_ylabel("Count")
ax4.set_title(f"ΔR² distribution  (mean={mean_delta:.3f},  95% CI [{ci_lo:.3f},{ci_hi:.3f}])")
fig4.tight_layout()
fig4.savefig(os.path.join(results_dir, f"ws_bs_delta_hist_{target_key}.png"), dpi=150)

plt.show()

print("\nDone. Figures and CSV written to:", results_dir)
