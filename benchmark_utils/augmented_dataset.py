from benchopt import BaseSolver, safe_import_context
from abc import abstractmethod, ABC
from benchmark_utils.transformation import channels_dropout, smooth_timemask
# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.utils import resample


class AugmentedBCISolver(BaseSolver, ABC):
    """Base class for solvers that use augmented data.

    This class implements some basic methods to run methods with various
    augmentation levels.
    """

    parameters = {
        "augmentation": [
            "SmoothTimeMask",
            "ChannelsDropout",
            "IdentityTransform",
        ],
    }

    @abstractmethod
    def set_objective(self, **objective_dict):
        """Set the objective information from Objective.get_objective.

        Objective
        ---------
        X: training data for the model
        y: training labels to train the model.
        sfreq: sampling frequency to allow filtering the data.
        """
        pass

    def run(self, n_augmentation):
        """Run the solver to evaluate it for a given number of augmentation.

        With this dataset, we consider that the performance curve is sampled
        for various number of augmentation applied to the dataset.
        """
        if self.augmentation == "ChannelsDropout":
            X, y = channels_dropout(
                self.X, self.y, n_augmentation=n_augmentation
            )

        elif self.augmentation == "SmoothTimeMask":
            X, y = smooth_timemask(
                self.X, self.y, n_augmentation=n_augmentation, sfreq=self.sfreq
            )
        elif self.augmentation == "Sampler":
            samples_list = [0.1, 0.25, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 20]
            n_samples = int(len(self.X) * samples_list[n_augmentation])
            X, y = resample(self.X, self.y,
                            n_samples=n_samples,
                            random_state=42)
        else:
            X = self.X
            y = self.y

        self.clf.fit(X, y)

    def get_next(self, n_iter):
        return n_iter + 1

    def get_result(self):
        """Return the model to `Objective.evaluate_result`.

        Result
        ------
        model: an instance of a fitted model.
            This model should have methods `score` and `predict`, that accept
            braindecode.WindowsDataset as input.
        """
        return dict(model=self.clf)
