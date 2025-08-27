import os
import numpy as np

# Path to the file
load_path = os.path.join(
    os.getcwd(),
    'eegfmri_data_07122025',
    'processed_data',
    'eeg_fmri_data.npz'
)

# Load file
data = np.load(load_path, allow_pickle=True)

# Extract subject IDs
subject_ids = data['subject_ids']

# Count unique subjects
unique_subs = np.unique(subject_ids)
print(f"Number of unique subjects: {len(unique_subs)}")
print("Subjects:", unique_subs)

