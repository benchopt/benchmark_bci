from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from numpy import multiply, array
    from skorch.helper import SliceDataset
    from braindecode.preprocessing import (
        preprocess,
        Preprocessor,
    )
    from braindecode.preprocessing import create_windows_from_events


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
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
    ]

    # Transform the data
    preprocess(dataset, preprocessors, n_jobs=n_jobs)

    return dataset


def windows_data(
    dataset,
    paradigm_name,
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
        mapping = {"left_hand": 1, "right_hand": 2}

    elif paradigm_name == "MotorImagery":
        mapping = {"left_hand": 1, "right_hand": 2, "feet": 4, "tongue": 3}

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
    )

    return windows_dataset, sfreq


def split_windows_train_test(data_subject_test, data_subject_train):
    """
    Split the window dataset into train and test sets.
    """
    # Converting the windows dataset into numpy arrays
    X_test = SliceDataset(data_subject_test, idx=0)
    y_test = array(list(SliceDataset(data_subject_test, idx=1)))-1

    X_train = SliceDataset(data_subject_train, idx=0)
    y_train = array(list(SliceDataset(data_subject_train, idx=1)))-1

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
