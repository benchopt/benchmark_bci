from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from moabb.datasets import fake
    from moabb.paradigms import MotorImagery
    from braindecode.datasets import create_from_X_y


class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.

    def get_data(self):

        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        dataset = fake.FakeDataset()
        paradigm = MotorImagery(n_classes=3)
        X, y, _ = paradigm.get_data(dataset=dataset,
                                    subjects=[1])
        sfreq = 1000
        dataset = create_from_X_y(X,
                                  y,
                                  drop_last_window=False,
                                  sfreq=sfreq)

        return dict(dataset=dataset,
                    paradigm_name='MotorImagery')
