import os
import mne
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


sub = 'sub-001'
ses = '001'
bld = '001'

eegfile = os.path.join(
        data_path, sub, f'ses-{ses}', 'eeg',
        f'{sub}_ses-{ses}_bld{bld}_eeg_Bergen_CWreg_filt_ICA_rej.set'
    )


# Load EEG
raw = mne.io.read_raw_eeglab(eegfile, preload=True)

# Extract events
event_label='T  1'
events, event_id = mne.events_from_annotations(raw)
if event_label not in event_id:
    raise ValueError(f"Event label '{event_label}' not found in the annotations.")

idx_T = events[events[:, 2] == event_id[event_label], 0]
idx_T_sec = raw.times[idx_T]

# Compute TFR
freqs = np.linspace(1, 40, 40)
decim = 10
n_cycles = freqs / 2
tfr = raw.compute_tfr(
    method='morlet',
    freqs=freqs,
    output='power',
    n_jobs=1,
    decim=decim,
    n_cycles=n_cycles,
    use_fft=True,
    zero_mean=True
)

# Print channel names
print(tfr.ch_names)

# Display TFR for a specific sensor Fz
tfr.plot(picks='Fz', baseline=(None, 0), mode='logratio', title='TFR for channel Fz')



