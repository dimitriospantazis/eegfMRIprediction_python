import os         
import numpy as np
from utils import compute_pre_event_tfr_segments

data_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025')

all_x = []
all_y_DAN = []
all_y_DMN = []
all_y_DNa = []
all_y_DNb = []
subject_ids = []
session_ids = []  # NEW: store the session number for each sample

subjects = [f"sub-{i:03d}" for i in [
    1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20,
    21, 23, 24, 25
]]

sessions = ['001', '002']
blocks = ['001', '002', '003', '004']

bin_size = 1.0
n_bins = 20

for sub in subjects:
    for ses in sessions:
        for bld in blocks:
            try:
                eegfile = os.path.join(
                    data_path, sub, f'ses-{ses}', 'eeg',
                    f'{sub}_ses-{ses}_bld{bld}_eeg_Bergen_CWreg_filt_ICA_rej.set'
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

                if not all(os.path.isfile(f) for f in [eegfile, danfile, dmnfile, dnafile, dnbfile]):
                    print(f'Skipping missing files for {sub}, ses-{ses}, bld{bld}')
                    continue

                # Load data
                x = compute_pre_event_tfr_segments(eegfile, bin_size=bin_size, n_bins=n_bins)
                y_dan = np.loadtxt(danfile).reshape(-1, 1)
                y_dmn = np.loadtxt(dmnfile).reshape(-1, 1)
                y_dna = np.loadtxt(dnafile).reshape(-1, 1)
                y_dnb = np.loadtxt(dnbfile).reshape(-1, 1)

                # Trim to shortest length across sources
                min_len = min(x.shape[0], y_dan.shape[0], y_dmn.shape[0], y_dna.shape[0], y_dnb.shape[0])
                x = x[:min_len]
                y_dan = y_dan[:min_len]
                y_dmn = y_dmn[:min_len]
                y_dna = y_dna[:min_len]
                y_dnb = y_dnb[:min_len]

                # Append
                all_x.append(x)
                all_y_DAN.append(y_dan)
                all_y_DMN.append(y_dmn)
                all_y_DNa.append(y_dna)
                all_y_DNb.append(y_dnb)
                subject_ids.extend([sub] * x.shape[0])
                session_ids.extend([ses] * x.shape[0])  # store session for each trial

            except Exception as e:
                print(f"Error in {sub}, ses-{ses}, bld{bld}: {e}")

# Concatenate all subjects
X = np.concatenate(all_x, axis=0)
Y_DAN = np.concatenate(all_y_DAN, axis=0)
Y_DMN = np.concatenate(all_y_DMN, axis=0)
Y_DNa = np.concatenate(all_y_DNa, axis=0)
Y_DNb = np.concatenate(all_y_DNb, axis=0)
subject_ids = np.array(subject_ids)
session_ids = np.array(session_ids)

# Relative time bins (closest to event = 0s)
bin_times = -np.arange(n_bins) * bin_size  # e.g., [0, -1, -2, ..., -19]s

print(f'Final shapes:\nX = {X.shape}\nY_DAN = {Y_DAN.shape}\nY_DMN = {Y_DMN.shape}\n'
      f'Y_DNa = {Y_DNa.shape}\nY_DNb = {Y_DNb.shape}\n'
      f'subject_ids = {subject_ids.shape}\nsession_ids = {session_ids.shape}')

# Save
processed_dir = os.path.join(data_path, 'processed_data')
os.makedirs(processed_dir, exist_ok=True)
save_path = os.path.join(processed_dir, 'eeg_fmri_data.npz')
np.savez_compressed(
    save_path,
    X=X,
    Y_DAN=Y_DAN,
    Y_DMN=Y_DMN,
    Y_DNa=Y_DNa,
    Y_DNb=Y_DNb,
    subject_ids=subject_ids,
    session_ids=session_ids,  # NEW
    bin_times=bin_times
)
print(f'Data saved to {save_path}')
