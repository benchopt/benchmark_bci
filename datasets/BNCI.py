from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from moabb.datasets import BNCI2014001
    from moabb.paradigms import LeftRightImagery, MotorImagery

# All datasets must be named `Dataset` and inherit from `BaseDataset`


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
        dataset = BNCI2014001()
        if self.paradigm_name == 'LeftRightImagery':

            paradigm = LeftRightImagery(fmin=8, fmax=35)
            # The dictionary defines the keyword arguments
            # for `Objective.set_data`
            return dict(dataset=dataset, paradigm=paradigm)

        else:
            paradigm = MotorImagery(n_classes=3)
            return dict(dataset=dataset, paradigm=paradigm)
