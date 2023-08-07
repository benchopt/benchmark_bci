from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from braindecode.datasets import MOABBDataset
    from benchmark_utils import windows_data


class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "BNCI"
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

        dataset_name = "BNCI2014001"
        data = MOABBDataset(dataset_name=dataset_name, subject_ids=None)

        dataset, sfreq = windows_data(data, self.paradigm_name)

        return dict(dataset=dataset, sfreq=sfreq)
