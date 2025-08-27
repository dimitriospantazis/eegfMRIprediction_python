import os
import numpy as np
import mne
import matplotlib.pyplot as plt

def plot_31ch_topomap(vals31, eegfile, title="Custom 31-channel values"):
    """
    Plot a topomap for a 31-dimensional vector using the first 31 EEG channels
    from an EEGLAB file.

    Parameters
    ----------
    vals31 : array-like, shape (31,)
        Values for the 31 EEG channels, in the correct order.
    eegfile : str
        Path to the EEGLAB .set file.
    title : str, optional
        Title for the plot.
    """
    vals31 = np.asarray(vals31, dtype=float)
    if vals31.shape[0] != 31:
        raise ValueError(f"Expected 31 values, got {vals31.shape[0]}.")

    # Load the EEG file
    raw = mne.io.read_raw_eeglab(eegfile, preload=False, verbose="ERROR")

    # Take the first 31 EEG channels
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')[:31]
    if len(picks) != 31:
        raise ValueError(f"File has {len(picks)} EEG channels; need 31.")

    info31 = mne.pick_info(raw.info, picks)

    # Create EvokedArray with a single time point
    data = vals31.reshape(31, 1)
    evoked = mne.EvokedArray(data, info31, tmin=0.0, comment=title)

    # Plot topomap
    fig = evoked.plot_topomap(
        times=[0.0],
        ch_type='eeg',
        sensors=True,
        contours=6,
        time_format='',
        scalings=dict(eeg=1.0),
        colorbar=True
    )
    plt.suptitle(title)
    plt.show()
    return fig

# ======================
# Example usage
# ======================
if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), 'eegfmri_data_07122025')
    example_sub = "sub-001"
    example_ses = "001"
    example_bld = "001"
    eegfile = os.path.join(
        data_path, example_sub, f'ses-{example_ses}', 'eeg',
        f'{example_sub}_ses-{example_ses}_bld{example_bld}_eeg_Bergen_CWreg_filt_ICA_rej.set'
    )

    vals31 = np.linspace(-1, 1, 31)
    plot_31ch_topomap(vals31, eegfile, title="Example plot")



