import os
import numpy as np
from utils import compute_pre_event_tfr_segments

data_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025')

all_x = []
all_y = []
subject_ids = []

subjects = [f"sub-{i:03d}" for i in [
    1, 2, 4, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20,
    21, 23, 24, 25
]]

sessions = ['001', '002']
blocks = ['001', '002', '003', '004']

for sub in subjects:
    for ses in sessions:
        for bld in blocks:
            try:
                eegfile = os.path.join(
                    data_path, sub, f'ses-{ses}', 'eeg',
                    f'{sub}_ses-{ses}_bld{bld}_eeg_Bergen_CWreg_filt_ICA_rej.set'
                )

                fmrifile = os.path.join(
                    data_path, sub, f'ses-{ses}', 'func',
                    f'ses-{ses}_{bld}_ArealMSHBM300.txt'
                )

                if not os.path.isfile(eegfile) or not os.path.isfile(fmrifile):
                    print(f'Skipping missing files for {sub}, ses-{ses}, bld{bld}')
                    continue

                x = compute_pre_event_tfr_segments(eegfile)  # shape: (n_events, n_channels, n_freqs, n_bins)
                y = np.loadtxt(fmrifile, delimiter=',')       # shape: (n_events, 300)

                if x.shape[0] != y.shape[0]:
                    # Correct for potential mismatch in number of events by removing the last event in x
                    x = x[:-1]
                    if x.shape[0] != y.shape[0]:
                        print(f'Shape mismatch for {sub}, ses-{ses}, bld{bld}: x={x.shape}, y={y.shape}')
                        continue

                all_x.append(x)
                all_y.append(y)
                subject_ids.extend([sub] * x.shape[0])

            except Exception as e:
                print(f"Error in {sub}, ses-{ses}, bld{bld}: {e}")



# Concatenate all subjects
X = np.concatenate(all_x, axis=0)         # shape: (N, C, F, B)
Y = np.concatenate(all_y, axis=0)         # shape: (N, 300)
subject_ids = np.array(subject_ids)       # shape: (N,)

print(f'Final shapes:\nX = {X.shape}\nY = {Y.shape}\nsubject_ids = {subject_ids.shape}')

# Create output directory if it doesn't exist
processed_dir = os.path.join(data_path, 'processed_data')
os.makedirs(processed_dir, exist_ok=True)

# File path to save
save_path = os.path.join(processed_dir, 'eeg_fmri_data.npz')

# Save as compressed NPZ file
np.savez_compressed(save_path, X=X, Y=Y, subject_ids=subject_ids)

print(f'Data saved to {save_path}')

