from benchopt import BaseSolver
from abc import abstractmethod, ABC

from benchmark_utils.transformation import channels_dropout, smooth_timemask


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
