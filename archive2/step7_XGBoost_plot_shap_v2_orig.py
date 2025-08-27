import os
import numpy as np

# ---- Headless backend (no GUI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
plt.ioff()
plt.show = lambda *a, **k: None  # in case helper calls plt.show()

# Your topomap helper (assumed importable)
from utils import plot_31ch_topomap

# =========================
# Config
# =========================
data_root   = os.path.join(os.getcwd(), "eegfmri_data_07122025")
results_dir = os.path.join(os.getcwd(), "results")
fig_root    = os.path.join(results_dir, "shap_topos")
os.makedirs(fig_root, exist_ok=True)

TARGET_KEYS = ["Y_DAN", "Y_DMN", "Y_DNa", "Y_DNb"]
TIME_AGG = "sum"   # or "mean"
ABS_CMAP = "viridis"
SIGNED_CMAP = "RdBu_r"

# Okabe–Ito (colorblind-safe) palette
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
# Stable band->color mapping (case-insensitive); falls back to palette order if band not listed
BAND_COLOR_MAP = {
    "delta": "#0072B2",  # blue
    "theta": "#E69F00",  # orange
    "alpha": "#009E73",  # bluish green
    "beta":  "#D55E00",  # vermillion
    "gamma": "#CC79A7",  # purple
}

# Optional markers to help distinguish in grayscale (cycled)
LINE_MARKERS = ["o", "s", "^", "D", "P", "X", "*", "v"]

example_sub = "sub-001"
example_ses = "001"
example_bld = "001"
eegfile = os.path.join(
    data_root, example_sub, f"ses-{example_ses}", "eeg",
    f"{example_sub}_ses-{example_ses}_bld{example_bld}_eeg_Bergen_CWreg_filt_ICA_rej.set"
)

# =========================
# Helpers
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def clean_band_names(band_arr):
    return [(b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b)) for b in band_arr]

def aggregate_time(arr_cbt, mode="sum"):
    """
    arr_cbt: (channels, bands, timebins)
    return (bands, channels) aggregated over time
    """
    if mode == "mean":
        per_band = arr_cbt.mean(axis=2)
    else:
        per_band = arr_cbt.sum(axis=2)
    return per_band.transpose(1, 0)  # (bands, channels)

def figure_from_return(ret):
    if ret is None:
        return plt.gcf()
    if hasattr(ret, "savefig"):
        return ret
    if isinstance(ret, (list, tuple)):
        for obj in ret:
            if hasattr(obj, "savefig"):
                return obj
    return plt.gcf()

def apply_clim_to_axis(ax, vmin, vmax):
    for im in ax.images:
        try: im.set_clim(vmin=vmin, vmax=vmax)
        except Exception: pass
    for coll in ax.collections:
        if hasattr(coll, "set_clim"):
            try: coll.set_clim(vmin=vmin, vmax=vmax)
            except Exception: pass
    for art in ax.artists:
        if hasattr(art, "set_clim"):
            try: art.set_clim(vmin=vmin, vmax=vmax)
            except Exception: pass

def call_topomap_on_ax(ax, vals, title, cmap=None, vmin=None, vmax=None):
    kwargs = {"title": title, "ax": ax, "colorbar": False}
    if cmap is not None: kwargs["cmap"] = cmap
    if vmin is not None: kwargs["vmin"] = vmin
    if vmax is not None: kwargs["vmax"] = vmax
    try:
        ret = plot_31ch_topomap(vals, eegfile, **kwargs)
    except TypeError:
        try:
            kwargs.pop("ax", None)
            ret = plot_31ch_topomap(vals, eegfile, **kwargs)
        except TypeError:
            ret = plot_31ch_topomap(vals, eegfile, title=title)
    if (vmin is not None) and (vmax is not None):
        apply_clim_to_axis(ax, vmin, vmax)
    return figure_from_return(ret)

def add_shared_cbar(fig, cax, cmap_name, vmin, vmax, signed=False, label=""):
    if signed:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    if label:
        cb.set_label(label)
    return cb

def save_fig(fig, out_base, dpi=300):
    ensure_dir(os.path.dirname(out_base))
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out_base}.png / .pdf")
    plt.close(fig)

def pick_band_color(bname, i):
    key = str(bname).strip().lower()
    return BAND_COLOR_MAP.get(key, OKABE_ITO[i % len(OKABE_ITO)])

def plot_time_series(per_band_time, bin_times, band_names, signed, out_base, title):
    """
    per_band_time : (n_bands, n_bins) averaged across channels
    bin_times     : (n_bins,) centers (negative, e.g., [-1,-3,...,-19])
    signed        : bool -> symmetric y-lims if True; start at 0 if False
    """
    # Convert to seconds before event (positive) and sort increasing
    t_sec = -np.asarray(bin_times, dtype=float)
    order = np.argsort(t_sec)
    t_sec = t_sec[order]
    data = per_band_time[:, order]

    # y-lims
    if signed:
        ymax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
        ymax = 1e-12 if ymax == 0 else ymax
        ymin, ymax = -ymax, ymax
        ylab = "Signed SHAP"
    else:
        ymax = np.nanmax(data) if np.isfinite(data).any() else 1.0
        ymax = 1e-12 if ymax == 0 else ymax
        ymin = 0.0
        ylab = "|SHAP|"

    # Build odd-second tick labels starting from 1
    max_t = float(np.nanmax(t_sec)) if t_sec.size else 0.0
    xticks = np.arange(1, int(np.floor(max_t)) + 1, 2)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    me = max(1, len(t_sec) // 10)  # marker spacing to avoid clutter

    for bi, bname in enumerate(band_names):
        color = pick_band_color(bname, bi)
        marker = LINE_MARKERS[bi % len(LINE_MARKERS)]
        ax.plot(t_sec, data[bi], lw=2.0, label=bname,
                color=color, marker=marker, markevery=me, ms=5)

    # Start axis at 0, label only odd seconds
    ax.set_xlim(0.0, max_t)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x)}" for x in xticks])

    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Time before event (s)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=False, ncol=len(band_names),
              loc="upper center", bbox_to_anchor=(0.5, -0.18))

    save_fig(fig, out_base)


# =========================
# Main
# =========================
for target_key in TARGET_KEYS:
    npz_path = os.path.join(results_dir, f"xgb_loso_r2_and_shap_{target_key}_0to20s.npz")
    if not os.path.exists(npz_path):
        print(f"[WARN] Missing NPZ for {target_key}: {npz_path}")
        continue

    D = np.load(npz_path, allow_pickle=True)
    band_names = clean_band_names(D["band_names"])
    imp_abs_cbb = D["imp_abs_cbb"]      # (channels, bands, time)
    imp_signed_cbb = D["imp_signed_cbb"]

    if "bin_times" in D.files:
        bin_times = D["bin_times"]
    else:
        n_bins = imp_abs_cbb.shape[-1]
        bin_times = -(np.arange(n_bins) * 2 + 1).astype(float)

    # ---------- Topomap grids (time aggregated) ----------
    per_band_abs    = aggregate_time(imp_abs_cbb, mode=TIME_AGG)    # (bands, channels)
    per_band_signed = aggregate_time(imp_signed_cbb, mode=TIME_AGG)

    vmax_abs = np.nanmax(per_band_abs) if np.isfinite(per_band_abs).any() else 1.0
    vmax_abs = 1e-12 if vmax_abs == 0 else vmax_abs
    vmin_abs = 0.0

    vmax_signed = np.nanmax(np.abs(per_band_signed)) if np.isfinite(per_band_signed).any() else 1.0
    vmax_signed = 1e-12 if vmax_signed == 0 else vmax_signed
    vmin_signed = -vmax_signed

    n_bands = len(band_names)

    # ABS grid
    fig = plt.figure(figsize=(3.4 * n_bands, 3.9))
    gs = gridspec.GridSpec(2, n_bands, height_ratios=[20, 1], hspace=0.25, wspace=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_bands)]
    cax = fig.add_subplot(gs[1, :])
    for i, (ax, bname) in enumerate(zip(axes, band_names)):
        vals = per_band_abs[i, :]
        call_topomap_on_ax(ax, vals, title=f"{bname}", cmap=ABS_CMAP, vmin=vmin_abs, vmax=vmax_abs)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{target_key} — abs SHAP ({TIME_AGG} over time)", y=0.98, fontsize=12)
    out_base = os.path.join(fig_root, target_key, f"{target_key}_absSHAP_{TIME_AGG}_grid")
    add_shared_cbar(fig, cax, ABS_CMAP, vmin_abs, vmax_abs, signed=False, label="|SHAP|")
    save_fig(fig, out_base)

    # SIGNED grid
    fig = plt.figure(figsize=(3.4 * n_bands, 3.9))
    gs = gridspec.GridSpec(2, n_bands, height_ratios=[20, 1], hspace=0.25, wspace=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_bands)]
    cax = fig.add_subplot(gs[1, :])
    for i, (ax, bname) in enumerate(zip(axes, band_names)):
        vals = per_band_signed[i, :]
        call_topomap_on_ax(ax, vals, title=f"{bname}", cmap=SIGNED_CMAP, vmin=vmin_signed, vmax=vmax_signed)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{target_key} — signed SHAP ({TIME_AGG} over time)", y=0.98, fontsize=12)
    out_base = os.path.join(fig_root, target_key, f"{target_key}_signedSHAP_{TIME_AGG}_grid")
    add_shared_cbar(fig, cax, SIGNED_CMAP, vmin_signed, vmax_signed, signed=True, label="signed SHAP")
    save_fig(fig, out_base)

    # ---------- Time-series (channel-averaged) ----------
    abs_bt    = imp_abs_cbb.mean(axis=0)     # (bands, time)
    signed_bt = imp_signed_cbb.mean(axis=0)  # (bands, time)

    out_base = os.path.join(fig_root, target_key, f"{target_key}_absSHAP_time_series_channelMean")
    plot_time_series(abs_bt, bin_times, band_names, signed=False,
                     out_base=out_base,
                     title=f"{target_key} — |SHAP| over time (channels mean)")

    out_base = os.path.join(fig_root, target_key, f"{target_key}_signedSHAP_time_series_channelMean")
    plot_time_series(signed_bt, bin_times, band_names, signed=True,
                     out_base=out_base,
                     title=f"{target_key} — signed SHAP over time (channels mean)")
# End (no plt.show())
