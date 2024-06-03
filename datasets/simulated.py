from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from braindecode.datasets import MOABBDataset
    from benchmark_utils import windows_data


_fakedataset_kwargs = {
    "n_sessions": 3,
    "n_runs": 2,
    "n_subjects": 3,
    "paradigm": "imagery",
    "duration": 3869,  # from bnci
    "sfreq": 250,
    "event_list": ("left_hand", "right_hand"),
    "channels": ("C5", "C3", "C1"),
    "annotations": True,
}


class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    def get_data(self):
        """Returns the data to be passed to Objective.set_data.

        Data
        ----
        Dataset: an instance of a braindecode.WindowsDataset
        sfreq: the sampling frequency of the data.
        """

        dataset_name = "FakeDataset"
        paradigm_name = "LeftRightImagery"
        data = MOABBDataset(
            dataset_name=dataset_name,
            subject_ids=None,
            dataset_kwargs=_fakedataset_kwargs,
        )

        dataset, sfreq = windows_data(data, paradigm_name)

        return dict(dataset=dataset, sfreq=sfreq, paradigm_name=paradigm_name)
