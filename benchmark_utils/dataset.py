from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import contextlib
    import io
    import os
    from pathlib import Path
    from pickle import load, dump
    from numpy import multiply
    from braindecode.preprocessing import (
        preprocess,
        Preprocessor,
    )
    from braindecode.preprocessing import create_windows_from_events, Pick
    from braindecode.datautil import load_concat_dataset
    from benchopt.config import get_setting
    from joblib import Memory
    from benchmark_utils import turn_off_warnings

    turn_off_warnings()


def rescaling(data, factor=1e6):
    return multiply(data, factor)


def pre_process_windows_dataset(
        dataset, low_cut_hz=4.0, high_cut_hz=38.0, factor=1e6, n_jobs=-1
):
    """
    Preprocess the window dataset.
        Function to apply preprocessing to the window (epoched) dataset.
        We proceed as follows:
        - Pick only EEG channels
        - Convert from V to uV
        - Bandpass filter

    Parameters:
    -----------
    dataset: WindowsDataset or BaseConcatDataset
        Dataset to use.
    low_cut_hz: float
        Low cut frequency for the bandpass filter.
    high_cut_hz: float
        High cut frequency for the bandpass filter.
    factor: float
        Factor to convert from V to uV.
    n_jobs: int
        Number of jobs to use for parallelization.
    Returns:
    --------
    dataset: WindowsDataset or BaseConcatDataset
        Preprocessed dataset.
    """
    # Parameters for exponential moving standardization
    preprocessors = [
        Pick(picks=['eeg']),
        # Keep EEG sensors
        Preprocessor(rescaling, factor=1e6),  # Convert from V to uV
        # Bandpass filter
        Preprocessor(
            "filter", l_freq=low_cut_hz, h_freq=high_cut_hz, verbose=False
        ),
    ]

    # Transform the data
    preprocess(dataset, preprocessors, n_jobs=n_jobs)

    return dataset


def windows_data(
        dataset,
        dataset_name,
        events_labels,
        paradigm_name,
        low_cut_hz=4.0,
        high_cut_hz=38.0,
        unit_factor=1e6,
        n_jobs=-1,
):
    """Create windows from the dataset.

    Parameters:
    -----------
    dataset: MOABBDataset
        Dataset to use.
    paradigm_name: str
        Name of the paradigm to use.
    dataset_name: str
        Name of the dataset.
    Returns:
    --------
    windows_dataset: WindowsDataset
        Dataset with windows.
    sfreq: float
        Sampling frequency of the dataset.
    """
    # Define mapping of classes to integers
    # We use two classes from the dataset
    # 1. left-hand vs right-hand motor imagery

    mem = Memory(get_setting("cache") or "__cache__", verbose=0)
    filename = f"{dataset_name}_dataset_{paradigm_name}"
    save_path = Path(mem.location) / filename
    save_obj = (
            Path(
                mem.location) / f"{filename}.pickle"
    )
    try:
        try:
            file = open(save_obj, "rb")
            windows_dataset = load(file)
        except Exception:
            if not save_path.exists():
                raise FileNotFoundError
            # Capturing verbose output
            f = io.StringIO()
            # Hacking way to capture verbose output
            with contextlib.redirect_stdout(f):
                windows_dataset = load_concat_dataset(
                    str(save_path.resolve()), preload=False, n_jobs=1
                )
        # Here, we decide that we will gonna use WindowsDataset only
        sfreq = dataset.datasets[0].raw.info["sfreq"]
        print(f"Using cached windows dataset {paradigm_name}.")
    except FileNotFoundError:
        print(f"Creating windows dataset {paradigm_name}.")
        dataset = pre_process_windows_dataset(
            dataset=dataset,
            low_cut_hz=low_cut_hz,
            high_cut_hz=high_cut_hz,
            factor=unit_factor,
            n_jobs=n_jobs,
        )
        # Extract sampling frequency, check that they are same in all datasets
        sfreq = dataset.datasets[0].raw.info["sfreq"]
        assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])

        extra_time_before, extra_time_after = (
            extra_time_seconds(dataset_name, sfreq))

        mapping_events = _fix_events_labels(events_labels)
        # Create windows using braindecode function for this.
        # It needs parameters to define how trials should be used.
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=extra_time_before,
            trial_stop_offset_samples=extra_time_after,
            preload=False,
            mapping=mapping_events,
            drop_last_window=True,
        )
        if not save_obj.exists():
            with open(save_obj, "wb") as file:
                dump(windows_dataset, file)

        if not save_path.exists():
            save_path.mkdir()
        windows_dataset.save(str(save_path.resolve()), overwrite=True)

    return windows_dataset, sfreq


def detect_if_cluster():
    """
    Utility function to detect if the code is running on a cluster or not.
    Returns:
    --------
    mne_path
    """
    if os.path.exists("/data/") and os.path.exists("/project/"):
        mne_path = Path("/data/")
    else:
        mne_path = None  # Path.home() / "mne_data/"
    # TODO: Make this for Jean Zay too.

    return mne_path


def extra_time_seconds(dataset_name, sfreq):
    """

    Parameters
    ----------
    dataset_name: str
        MOABB Dataset name

    sfreq: int
        dataset frequency


    Returns
    -------
    trial_start_offset_samples: int
        extra time before the windows in seconds
    trial_stop_offset_samples: int
        extra time after the windows in seconds
    """
    trial_start_offset_samples = 0
    trial_stop_offset_samples = 0
    if dataset_name == "BCNI2014_1":
        trial_start_offset_seconds = -0.5
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
        trial_stop_offset_samples = 0

        return trial_start_offset_samples, trial_stop_offset_samples
    else:
        return trial_start_offset_samples, trial_stop_offset_samples


def _fix_events_labels(events_label):
    """
    Parameters
    ----------
    events_label: dict
        Dictionary with events labels
        Example: {'left_hand': 1, 'right_hand': 2}

    Returns
    --------
    fixed_events_label: dict
        Dictionary with fixed events labels
        Example: {'left_hand': 0, 'right_hand': 1}
    """
    minimum = min(events_label.values())
    if minimum == 0:
        return events_label
    else:
        return {k: v - minimum for k, v in events_label.items()}
