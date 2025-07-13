import mne
import numpy as np

def compute_pre_event_tfr_segments(eegfile, event_label='T  1', bin_size=2.0, n_bins=5,
                                   freqs=np.linspace(1, 40, 40), decim=10):
    """
    Compute multiple pre-event TFR segments of fixed size (bin_size) before each event.

    Parameters
    ----------
    eegfile : str
        Path to the EEGLAB .set file.
    event_label : str
        Event label to extract (default 'T  1').
    bin_size : float
        Duration in seconds of each time bin (default 2.0).
    n_bins : int
        Number of time bins to extract before each event (default 5).
    freqs : array-like
        Frequencies to compute TFR (default 1–40 Hz, 1 Hz step).
    decim : int
        Decimation factor for TFR computation (default 10).

    Returns
    -------
    tfr_segments : np.ndarray
        Array of shape (n_events, n_channels, n_freqs, n_bins), where each entry is
        the average power in a time bin before the event.
    """
    # Load EEG
    raw = mne.io.read_raw_eeglab(eegfile, preload=True)

    # Extract events
    events, event_id = mne.events_from_annotations(raw)
    if event_label not in event_id:
        raise ValueError(f"Event label '{event_label}' not found in the annotations.")
    
    idx_T = events[events[:, 2] == event_id[event_label], 0]
    idx_T_sec = raw.times[idx_T]

    # Compute TFR
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

    sfreq = tfr.info['sfreq']
    times = tfr.times
    bin_samples = int(bin_size * sfreq)
    total_samples = bin_samples * n_bins

    # Store pre-event segments
    tfr_segments = []

    for ev_time in idx_T_sec:
        end_idx = np.argmin(np.abs(times - ev_time))
        start_idx = end_idx - total_samples

        # I want an error if out of bounds
        if start_idx < 0 or end_idx > len(times):
            raise ValueError(f"Event time {ev_time} is out of bounds for TFR data.")    

        bins = []
        for b in range(n_bins):
            b_start = start_idx + b * bin_samples
            b_end = b_start + bin_samples
            segment_bin = tfr.data[:, :, b_start:b_end].mean(axis=2)  # (n_channels, n_freqs)
            bins.append(segment_bin)

        # Stack bins along new axis → (n_channels, n_freqs, n_bins)
        bins = np.stack(bins, axis=-1)
        tfr_segments.append(bins)

    # Final shape: (n_events, n_channels, n_freqs, n_bins)
    tfr_segments = np.array(tfr_segments)
    return tfr_segments




