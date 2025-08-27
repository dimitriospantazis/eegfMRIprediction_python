import os 
import numpy as np
from utils import compute_pre_event_tfr_segments

data_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025')

all_x = []
all_y = []
all_y_DAN = []
all_y_DMN = []
all_y_DNa = []
all_y_DNb = []
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
                danfile = os.path.join(
                    data_path, sub, f'ses-{ses}', 'func',
                    f'DAN_ts_MSHBM_{sub}_bld{bld}_highpass.txt'
                )
                dmnfile = os.path.join(
                    data_path, sub, f'ses-{ses}', 'func',
                    f'DMN_ts_MSHBM_{sub}_bld{bld}_highpass.txt'
                )
                dnafile = os.path.join(
                    data_path, sub, f'ses-{ses}', 'func',
                    f'DNa_ts_MSHBM_{sub}_bld{bld}_highpass.txt'
                )
                dnbfile = os.path.join(
                    data_path, sub, f'ses-{ses}', 'func',
                    f'DNb_ts_MSHBM_{sub}_bld{bld}_highpass.txt'
                )

                if not all(os.path.isfile(f) for f in [eegfile, fmrifile, danfile, dmnfile, dnafile, dnbfile]):
                    print(f'Skipping missing files for {sub}, ses-{ses}, bld{bld}')
                    continue

                # Load data
                x = compute_pre_event_tfr_segments(eegfile, bin_size=1.0, n_bins=20)  # shape: (n_events, n_channels, n_freqs, n_bins)
                y = np.loadtxt(fmrifile, delimiter=',')       # shape: (n_events, 300)
                y_dan = np.loadtxt(danfile).reshape(-1, 1)    # reshape to (n_events, 1)
                y_dmn = np.loadtxt(dmnfile).reshape(-1, 1)
                y_dna = np.loadtxt(dnafile).reshape(-1, 1)
                y_dnb = np.loadtxt(dnbfile).reshape(-1, 1)

                if x.shape[0] != y.shape[0]:
                    # Correct mismatch by trimming x (and others if needed)
                    x = x[:y.shape[0]]
                    y_dan = y_dan[:y.shape[0]]
                    y_dmn = y_dmn[:y.shape[0]]
                    y_dna = y_dna[:y.shape[0]]
                    y_dnb = y_dnb[:y.shape[0]]
                    if x.shape[0] != y.shape[0]:
                        print(f'Shape mismatch for {sub}, ses-{ses}, bld{bld}: x={x.shape}, y={y.shape}')
                        continue

                # Append
                all_x.append(x)
                all_y.append(y)
                all_y_DAN.append(y_dan)
                all_y_DMN.append(y_dmn)
                all_y_DNa.append(y_dna)
                all_y_DNb.append(y_dnb)
                subject_ids.extend([sub] * x.shape[0])

            except Exception as e:
                print(f"Error in {sub}, ses-{ses}, bld{bld}: {e}")

# Concatenate all subjects
X = np.concatenate(all_x, axis=0)             # shape: (N, C, F, B)
Y = np.concatenate(all_y, axis=0)             # shape: (N, 300)
Y_DAN = np.concatenate(all_y_DAN, axis=0)     # shape: (N, 1)
Y_DMN = np.concatenate(all_y_DMN, axis=0)     # shape: (N, 1)
Y_DNa = np.concatenate(all_y_DNa, axis=0)     # shape: (N, 1)
Y_DNb = np.concatenate(all_y_DNb, axis=0)     # shape: (N, 1)
subject_ids = np.array(subject_ids)           # shape: (N,)

print(f'Final shapes:\nX = {X.shape}\nY = {Y.shape}\nY_DAN = {Y_DAN.shape}\nY_DMN = {Y_DMN.shape}\nY_DNa = {Y_DNa.shape}\nY_DNb = {Y_DNb.shape}\nsubject_ids = {subject_ids.shape}')

# Create output directory if it doesn't exist
processed_dir = os.path.join(data_path, 'processed_data')
os.makedirs(processed_dir, exist_ok=True)

# Save as compressed NPZ file
save_path = os.path.join(processed_dir, 'eeg_fmri_data.npz')
np.savez_compressed(
    save_path, 
    X=X, 
    Y=Y, 
    Y_DAN=Y_DAN, 
    Y_DMN=Y_DMN, 
    Y_DNa=Y_DNa, 
    Y_DNb=Y_DNb, 
    subject_ids=subject_ids
)

print(f'Data saved to {save_path}')

