# assumes: r_maps (n_subj, n_bins, n_freqs) and subjects (n_subj,) are in memory
import numpy as np, matplotlib.pyplot as plt

n_subj = r_maps.shape[0]
labels = [str(s) for s in (subjects if 'subjects' in locals() else range(n_subj))]
v = np.nanmax(np.abs(r_maps))
ncols = min(6, n_subj); nrows = int(np.ceil(n_subj / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.1, nrows*1.9), sharex=True, sharey=True)
axes = np.atleast_2d(axes)
im = None
for i, ax in enumerate(axes.ravel()):
    if i < n_subj:
        im = ax.imshow(r_maps[i], aspect='auto', origin='upper', cmap='RdBu_r', vmin=-v, vmax=v)
        ax.set_title(labels[i], fontsize=8); ax.axis('off')
    else:
        ax.axis('off')
fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.04, fraction=0.04, label='r')
fig.suptitle('Per-subject correlation maps', y=0.995, fontsize=11)
fig.tight_layout(rect=[0,0,1,0.97])
plt.show()

# (optional) quick consistency metric: corr of each subject map with group-mean map
gm = np.nanmean(r_maps, axis=0)
to_group = [np.corrcoef(r_maps[i].ravel(), gm.ravel())[0,1] for i in range(n_subj)]
plt.figure(figsize=(max(4, n_subj*0.25), 2.4))
plt.bar(range(n_subj), to_group); plt.ylim(-1, 1)
plt.xticks(range(n_subj), labels, rotation=60, ha='right', fontsize=8)
plt.ylabel('corr to group mean'); plt.title('Subject consistency (higher = more similar)')
plt.tight_layout(); plt.show()

