from benchopt import BaseDataset


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.


class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "BNCI"

    parameters = {'paradigm_name': ('LeftRightImagery', 'MotorImagery')}
    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.

    def get_data(self):

        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        dataset_name = "BNCI2014001"
        return dict(dataset_name=dataset_name,
                    paradigm_name=self.paradigm_name)
