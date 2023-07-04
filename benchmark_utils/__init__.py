# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax

from benchopt import safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from braindecode.preprocessing import (
        preprocess,
        Preprocessor,
    )
    from braindecode.preprocessing import create_windows_from_events
    from braindecode.augmentation import ChannelsDropout, SmoothTimeMask
    import torch


def list_train_test(test, list):
    list_test = [list[test]]
    list_train = list[:test] + list[test + 1:]
    return list_test, list_train


def transformX_moabb(X):
    """Transform X from moabb to numpy array.
    Parameters:
    -----------
    X: ndarray
        Data to transform.
    Returns:
    --------
    X0: ndarray
        Transformed data.
    """
    X0 = []
    for i in range(len(X)):
        X0.append(X[i])
    X0 = np.array(X0)
    return X0


def windows_data(dataset, paradigm_name):
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

    low_cut_hz = 4.0  # low cut frequency for filtering
    high_cut_hz = 38.0  # high cut frequency for filtering
    # Parameters for exponential moving standardization

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        # Keep EEG sensors
        Preprocessor(
            lambda data, factor: np.multiply(data, factor),
            # Convert from V to uV
            factor=1e6,
        ),
        # Bandpass filter
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
    ]

    # Transform the data
    preprocess(dataset, preprocessors, n_jobs=-1)
    trial_start_offset_seconds = -0.5
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this.
    # It needs parameters to define how
    # trials should be used.

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
        mapping=mapping,
    )

    return (windows_dataset, sfreq)


def channels_dropout(X, y, n_augmentation):
    seed = 20200220
    transform = ChannelsDropout(probability=0.5, random_state=seed)
    X_augm = transformX_moabb(X)
    y_augm = y
    for i in range(n_augmentation):
        X_tr, _ = transform.operation(
            torch.as_tensor(X).float(), None, p_drop=0.2
        )

        X_tr = X_tr.numpy()
        X_augm = np.concatenate((X_augm, X_tr))
        y_augm = np.concatenate((y_augm, y))

    return (X_augm, y_augm)


def smooth_timemask(X, y, n_augmentation, sfreq):
    second = 0.1
    seed = 20200220

    transform = SmoothTimeMask(
        probability=0.5,
        mask_len_samples=int(sfreq * second),
        random_state=seed,
    )

    X_torch = torch.as_tensor(X).float()
    y_torch = torch.as_tensor(y).float()
    param_augm = transform.get_augmentation_params(X_torch, y_torch)
    mls = param_augm["mask_len_samples"]
    msps = param_augm["mask_start_per_sample"]

    X_augm = transformX_moabb(X)
    y_augm = y

    for i in range(n_augmentation):
        X_tr, _ = transform.operation(
            X_torch, None, mask_len_samples=mls, mask_start_per_sample=msps
        )

        X_tr = X_tr.numpy()
        X_augm = np.concatenate((X_augm, X_tr))
        y_augm = np.concatenate((y_augm, y))

    return (X_augm, y_augm)
