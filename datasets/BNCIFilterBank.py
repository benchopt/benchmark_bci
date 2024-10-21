from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from braindecode.datasets import MOABBDataset
    from braindecode.preprocessing import create_windows_from_events, Pick

    from moabb.utils import set_download_dir
    from benchmark_utils import detect_if_cluster
    from benchmark_utils.dataset import rescaling
    from braindecode.preprocessing import (
        preprocess,
        Preprocessor,
        filterbank
    )


class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "BNCIFilterBank"
    parameters = {
        'paradigm_name': ('MotorImagery', 'LeftRightImagery')
    }

    def get_data(self):
        """Returns the data to be passed to Objective.set_data.

        Data
        ----
        Dataset: an instance of a braindecode.WindowsDataset
        sfreq: the sampling frequency of the data.
        """
        running_cluster = detect_if_cluster()
        if running_cluster is not None:
            set_download_dir(running_cluster)

        dataset_name = "BNCI2014_001"
        data = MOABBDataset(dataset_name=dataset_name, subject_ids=[1])

        dataset, sfreq = windows_data_filter(data, self.paradigm_name)

        print(dataset[0][0].shape)

        return dict(dataset=dataset, sfreq=sfreq,
                    paradigm_name=self.paradigm_name,
                    dataset_name=dataset_name)




def windows_data_filter(
    dataset,
    paradigm_name,
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
    mapping = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}

    print(f"Creating windows dataset {paradigm_name}.")

    frequency_bands = [(i, i+4) for i in range(4, 40, 4)]

    preprocessors = [
        Pick(picks=['eeg']),
        # Keep EEG sensors
        Preprocessor(rescaling, factor=1e-6),  # Convert from V to uV
        # Bandpass filter
        Preprocessor(fn=filterbank, frequency_bands=frequency_bands, drop_original_signals=True, apply_on_array=False),
    ]

    # Transform the data
    preprocess(dataset, preprocessors, n_jobs=1)
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    # Create windows using braindecode function for this.
    # It needs parameters to define how trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        preload=True,
        mapping=mapping,
        drop_bad_windows=True,
        drop_last_window=True,
    )

    return windows_dataset, sfreq
