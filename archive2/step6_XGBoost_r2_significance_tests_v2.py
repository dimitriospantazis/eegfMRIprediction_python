import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =========================
# Paths & config
# =========================
data_root   = os.path.join(os.getcwd(), "eegfmri_data_07122025")
results_dir = os.path.join(os.getcwd(), "results")
fig_dir     = os.path.join(results_dir, "figs_r2")
os.makedirs(fig_dir, exist_ok=True)

TARGET_KEYS = ["Y_DAN", "Y_DMN", "Y_DNa", "Y_DNb"]
FILE_TMPL = "xgb_loso_r2_and_shap_{key}_0to20s.npz"

# Save CSV/TSV?
WRITE_CSV = True
csv_table_path    = os.path.join(results_dir, "loso_inference_table_0to20s.tsv")
csv_long_r2_path  = os.path.join(results_dir, "loso_fold_r2_long_0to20s.tsv")

# Figure style
sns.set(context="paper", style="whitegrid", font_scale=1.2)
FIGSIZE = (8.0, 4.6)
DPI = 300

# Names & colorblind-safe palette (Okabe–Ito)
NAME_MAP = {"Y_DAN": "DAN", "Y_DMN": "DMN", "Y_DNa": "DNa", "Y_DNb": "DNb"}
PALETTE  = {
    "Y_DAN": "#0072B2",  # blue
    "Y_DMN": "#E69F00",  # orange
    "Y_DNa": "#009E73",  # bluish green
    "Y_DNb": "#D55E00",  # vermillion
}

# =========================
# Helpers
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def sign_test_p(count_pos, n):
    # one-sided P(X >= count_pos) for X~Bin(n, 0.5)
    return float(np.sum(stats.binom.pmf(np.arange(count_pos, n + 1), n, 0.5)))

def ci95_t(mean, sd, n):
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    return (mean - tcrit * se, mean + tcrit * se)

def format_ci(ci):
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"

def load_target_npz(key):
    path = os.path.join(results_dir, FILE_TMPL.format(key=key))
    if not os.path.exists(path):
        print(f"[WARN] Missing file for {key}: {path}")
        return None
    return np.load(path, allow_pickle=True)

def r2_stats(fold_r2):
    n = fold_r2.size
    mean = fold_r2.mean()
    sd   = fold_r2.std(ddof=1)
    ci   = ci95_t(mean, sd, n)
    median = float(np.median(fold_r2))
    positives = int((fold_r2 > 0).sum())
    t_res = stats.ttest_1samp(fold_r2, 0.0, alternative="greater")
    try:
        w_res = stats.wilcoxon(fold_r2, alternative="greater", zero_method="wilcox")
    except Exception:
        w_res = type("obj",(object,),{"statistic": np.nan, "pvalue": np.nan})()
    p_sign = sign_test_p(positives, n)
    return dict(n=n, mean=mean, sd=sd, ci=ci, median=median,
                positives=positives, t_stat=t_res.statistic, t_p=t_res.pvalue,
                w_V=w_res.statistic, w_p=w_res.pvalue, sign_p=p_sign)

def common_ylim(all_values, pad=0.02):
    ymin = min(np.min(v) for v in all_values if v.size > 0)
    ymax = max(np.max(v) for v in all_values if v.size > 0)
    ymin = min(0.0, ymin)
    ymax = max(0.0, ymax)
    span = ymax - ymin
    return (ymin - pad*span, ymax + pad*span)

def savefig(fig, path_base):
    ensure_dir(os.path.dirname(path_base))
    for ext in (".png", ".pdf"):
        out = path_base + ext
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)

# =========================
# Load, compute stats, collect data
# =========================
rows = []
fold_r2_dict = {}
fold_subjects_dict = {}

for key in TARGET_KEYS:
    D = load_target_npz(key)
    if D is None:
        continue
    fold_r2 = np.array(D["fold_r2"], dtype=float)
    fold_subjects = D["fold_subjects"] if "fold_subjects" in D.files else np.array([f"{i+1:02d}" for i in range(len(fold_r2))])
    s = r2_stats(fold_r2)

    rows.append(dict(
        target_key=key, n=s["n"],
        mean_r2=s["mean"], sd_r2=s["sd"], ci_low=s["ci"][0], ci_high=s["ci"][1],
        median_r2=s["median"], positives=s["positives"],
        t_stat=s["t_stat"], t_p=s["t_p"], wilcoxon_V=s["w_V"], wilcoxon_p=s["w_p"],
        sign_p=s["sign_p"]
    ))
    fold_r2_dict[key] = fold_r2
    fold_subjects_dict[key] = np.array(fold_subjects)

# =========================
# Print table (console) and write TSVs
# =========================
if rows:
    hdr = ("Target  N  MeanR2  [95% CI]        Median  +R2    t(>0) p        Wilcoxon V  p         Sign p")
    print(hdr)
    for r in rows:
        print(f"{NAME_MAP.get(r['target_key'], r['target_key']):6s} {r['n']:2d}  {r['mean_r2']:.3f}  "
              f"[{r['ci_low']:.3f},{r['ci_high']:.3f}]  "
              f"{r['median_r2']:.3f}  {r['positives']:2d}   "
              f"{r['t_stat']:6.2f} {r['t_p']:.2e}   "
              f"{str(r['wilcoxon_V']):>8s}  {r['wilcoxon_p']:.2e}   "
              f"{r['sign_p']:.2e}")

if WRITE_CSV and rows:
    cols = ["target_key","n","mean_r2","sd_r2","ci_low","ci_high","median_r2",
            "positives","t_stat","t_p","wilcoxon_V","wilcoxon_p","sign_p"]
    with open(csv_table_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\nSaved stats table: {csv_table_path}")

    with open(csv_long_r2_path, "w") as f:
        f.write("target_key\tsubject\tfold_r2\n")
        for key, vals in fold_r2_dict.items():
            subs = fold_subjects_dict[key]
            for s, v in zip(subs, vals):
                f.write(f"{NAME_MAP.get(key,key)}\t{s}\t{v:.6f}\n")
    print(f"Saved long-form fold R^2: {csv_long_r2_path}")

# =========================
# Single figure: violin + strip (mean ± 95% CI overlay)
# =========================
if not fold_r2_dict:
    raise SystemExit("No target files found; nothing to plot.")

# Prepare stacked data
stack_vals, stack_keys = [], []
present_keys = [k for k in TARGET_KEYS if k in fold_r2_dict]
order = [NAME_MAP.get(k, k) for k in present_keys]
palette = [PALETTE.get(k, "#333333") for k in present_keys]

for key in present_keys:
    vals = fold_r2_dict[key]
    stack_vals.extend(list(vals))
    stack_keys.extend([NAME_MAP.get(key, key)] * len(vals))

fig, ax = plt.subplots(figsize=FIGSIZE)

sns.violinplot(x=stack_keys, y=stack_vals, order=order, palette=palette,
               inner=None, cut=0, ax=ax, linewidth=0.8)
sns.stripplot(x=stack_keys, y=stack_vals, order=order, color="#222222",
              size=3.6, alpha=0.65, jitter=0.15, ax=ax)

# overlay mean ± 95% CI per group as diamonds with error bars
for i, key in enumerate(present_keys):
    vals = fold_r2_dict[key]
    s = r2_stats(vals)
    ax.errorbar(i, s["mean"],
                yerr=[[s["mean"] - s["ci"][0]], [s["ci"][1] - s["mean"]]],
                fmt="D", color="white", mec="black", ms=6, capsize=4, lw=1.5, zorder=5)

# cosmetics
ylims = common_ylim(list(fold_r2_dict.values()), pad=0.06)
ax.axhline(0, color="#555555", lw=1, ls="--")
ax.set_xlabel("")
ax.set_ylabel(r"$R^2$")
ax.set_ylim(*ylims)
ax.set_title("LOSO performance by target (subject-level)")
sns.despine(fig=fig)

savefig(fig, os.path.join(fig_dir, "violin_strip_r2_by_target"))
