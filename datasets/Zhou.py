from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    from braindecode.datasets import MOABBDataset
    from benchmark_utils import windows_data


class Dataset(BaseDataset):

    name = "Zhou"

    def get_data(self):
        """Returns the data to be passed to Objective.set_data.

        Data
        ----
        Dataset: an instance of a braindecode.WindowsDataset
        sfreq: the sampling frequency of the data.
        """

        dataset_name = "Zhou2016"
        paradigm_name = "LeftRightImagery"
        data = MOABBDataset(dataset_name=dataset_name, subject_ids=None)

        dataset, sfreq = windows_data(data, paradigm_name)

        return dict(dataset=dataset, sfreq=sfreq)
