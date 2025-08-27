import os
import numpy as np
from scipy import stats

# =========================
# Paths & config
# =========================
data_root = os.path.join(os.getcwd(), "eegfmri_data_07122025")
results_dir = os.path.join(os.getcwd(), "results")
TARGET_KEYS = ["Y_DAN", "Y_DMN", "Y_DNa", "Y_DNb"]
FILE_TMPL = "xgb_loso_r2_and_shap_{key}_0to20s.npz"

# Save CSV?
WRITE_CSV = True
csv_path = os.path.join(results_dir, "loso_inference_table_0to20s.tsv")

# =========================
# Helpers
# =========================
def sign_test_p(count_pos, n):
    # one-sided: P(X >= count_pos) for X~Bin(n, 0.5)
    return sum(stats.binom.pmf(np.arange(count_pos, n + 1), n, 0.5))

def ci95_t(mean, sd, n):
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, df=n-1)
    return (mean - tcrit * se, mean + tcrit * se)

# =========================
# Load, test, and print
# =========================
rows = []
for key in TARGET_KEYS:
    path = os.path.join(results_dir, FILE_TMPL.format(key=key))
    if not os.path.exists(path):
        print(f"Missing file for {key}: {path}")
        continue

    D = np.load(path, allow_pickle=True)
    fold_r2 = np.array(D["fold_r2"], dtype=float)
    overall_r2 = float(D["overall_r2"])
    n = fold_r2.size
    mean = fold_r2.mean()
    sd   = fold_r2.std(ddof=1)
    ci   = ci95_t(mean, sd, n)
    median = np.median(fold_r2)
    positives = int((fold_r2 > 0).sum())

    # One-sample tests vs 0 (subjects as units)
    t_res = stats.ttest_1samp(fold_r2, 0.0, alternative="greater")
    try:
        w_res = stats.wilcoxon(fold_r2, alternative="greater", zero_method="wilcox")
    except Exception:
        # Fallback if all values positive but identical (degenerate): set to NaN
        w_res = type("obj",(object,),{"statistic": np.nan, "pvalue": np.nan})()
    p_sign = sign_test_p(positives, n)

    rows.append(dict(
        target_key=key, n=n,
        mean_r2=mean, sd_r2=sd, ci_low=ci[0], ci_high=ci[1], median_r2=median,
        positives=positives, overall_r2=overall_r2,
        t_stat=t_res.statistic, t_p=t_res.pvalue,
        wilcoxon_V=w_res.statistic, wilcoxon_p=w_res.pvalue,
        sign_p=p_sign
    ))

# Pretty print table
if rows:
    hdr = ("Target  N  MeanR2  [95% CI]        Median  +R2  OverallR2  "
           "t(>0) p        Wilcoxon V  p         Sign p")
    print(hdr)
    for r in rows:
        print(f"{r['target_key']:6s} {r['n']:2d}  {r['mean_r2']:.3f}  "
              f"[{r['ci_low']:.3f},{r['ci_high']:.3f}]  "
              f"{r['median_r2']:.3f}  {r['positives']:2d}   {r['overall_r2']:.3f}   "
              f"{r['t_stat']:6.2f} {r['t_p']:.2e}   "
              f"{r['wilcoxon_V']!s:>8s}  {r['wilcoxon_p']:.2e}   "
              f"{r['sign_p']:.2e}")

# Write CSV/TSV for paper tables
if WRITE_CSV and rows:
    cols = ["target_key","n","mean_r2","sd_r2","ci_low","ci_high","median_r2",
            "positives","overall_r2","t_stat","t_p","wilcoxon_V","wilcoxon_p","sign_p"]
    with open(csv_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\nSaved stats table: {csv_path}")


