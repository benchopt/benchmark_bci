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
    from braindecode.preprocessing import create_windows_from_events
    from braindecode.datautil import load_concat_dataset
    from benchopt.config import get_setting
    from joblib import Memory
    from benchmark_utils import turn_off_warnings

    turn_off_warnings()


def pre_process_windows_dataset(
    dataset, low_cut_hz=4.0, high_cut_hz=38.0, factor=1e6, n_jobs=20
):
    """
    Preprocess the window dataset.
        Function to apply preprocessing to the window (epoched) dataset.
        We proceed as follows:
        - Pick only EEG channels
        - Convert from V to uV
        - Bandpass filter
        - Apply exponential moving standardization
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
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        # Keep EEG sensors
        Preprocessor(
            lambda data, factor: multiply(data, factor),
            # Convert from V to uV
            factor=factor,
        ),
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
    paradigm_name,
    dataset_name,
    trial_start_offset_seconds=-0.5,
    low_cut_hz=4.0,
    high_cut_hz=38.0,
    factor=1e6,
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

    if paradigm_name == "LeftRightImagery":
        mapping = {"left_hand": 0, "right_hand": 1}  # Fix this later

    elif paradigm_name == "MotorImagery":
        mapping = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}

    mem = Memory(get_setting("cache") or "__cache__", verbose=0)

    save_path = Path(mem.location) / f"{dataset_name}_dataset_{paradigm_name}"
    save_obj = (
        Path(mem.location) / f"{dataset_name}_dataset_{paradigm_name}.pickle"
    )
    try:
        raise FileNotFoundError
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

        sfreq = windows_dataset.datasets[0].windows.info["sfreq"]
        print(f"Using cached windows dataset {paradigm_name}.")
    except FileNotFoundError:
        print(f"Creating windows dataset {paradigm_name}.")
        dataset = pre_process_windows_dataset(
            dataset,
            low_cut_hz=low_cut_hz,
            high_cut_hz=high_cut_hz,
            factor=factor,
            n_jobs=n_jobs,
        )

        # Extract sampling frequency, check that they are same in all datasets
        sfreq = dataset.datasets[0].raw.info["sfreq"]
        assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

        # Create windows using braindecode function for this.
        # It needs parameters to define how trials should be used.
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True,
            mapping=mapping,
            drop_bad_windows=True,
            drop_last_window=True,
        )

        if not save_obj.exists():
            with open(save_obj, "wb") as file:
                dump(windows_dataset, file)

        #if not save_path.exists():
        #    save_path.mkdir()
        #windows_dataset.save(str(save_path.resolve()), overwrite=True)

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
