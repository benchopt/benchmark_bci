from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from braindecode.datasets import MOABBDataset
    from benchmark_utils import windows_data


class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "simulated"
    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.

    def get_data(self):

        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        dataset_name = "FakeDataset"
        data = MOABBDataset(dataset_name=dataset_name,
                            subject_ids=None,
                            dataset_kwargs={"event_list": ["left_hand", "right_hand"],
                                            "paradigm": "imagery"})

        dataset, sfreq = windows_data(data, 'LeftRightImagery')

        return dict(dataset=dataset,
                    paradigm_name='LeftRightImagery',
                    sfreq=sfreq)
