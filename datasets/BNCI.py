from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from braindecode.datasets import MOABBDataset
    from moabb.utils import set_download_dir
    from benchmark_utils import windows_data, detect_if_cluster


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
        running_cluster = detect_if_cluster()
        if running_cluster is not None:
            set_download_dir(running_cluster)

        dataset_name = "BNCI2014_001"
        data = MOABBDataset(dataset_name=dataset_name, subject_ids=None)

        dataset, sfreq = windows_data(data, self.paradigm_name, dataset_name)

        return dict(dataset=dataset, sfreq=sfreq,
                    paradigm_name=self.paradigm_name,
                    dataset_name=dataset_name)
