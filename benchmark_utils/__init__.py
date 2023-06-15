# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax

import numpy as np
from braindecode.preprocessing import (
    exponential_moving_standardize,
    preprocess,
    Preprocessor,
)
from braindecode.preprocessing import create_windows_from_events


def flatten(liste):
    return [item for sub_list in liste for item in sub_list]


def list_train_test(test, list):
    list_test = [list[test]]
    list_train = list[:test] + list[test+1:]
    return list_test, list_train


def windows_data(dataset, paradigm_name):

    # défintion du paradigm
    # if paradigm_name == 'LeftRightImagery':
    mapping = {'left_hand': 1, 'right_hand': 2}

    low_cut_hz = 4.0  # low cut frequency for filtering
    high_cut_hz = 38.0  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        # Keep EEG sensors
        Preprocessor(
            lambda data, factor: np.multiply(data, factor),
            # Convert from V to uV
            factor=1e6,
        ),
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
        # Bandpass filter
        Preprocessor(
            exponential_moving_standardize,
            # Exponential moving standardization
            factor_new=factor_new,
            init_block_size=init_block_size,
        ),
    ]

    # Transform the data
    preprocess(dataset, preprocessors)
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
        preload=True, mapping=mapping
    )

    return windows_dataset
