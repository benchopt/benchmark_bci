from benchopt import BaseSolver, safe_import_context
from abc import abstractmethod, ABC

from benchmark_utils.transformation import (
    channels_dropout,
    smooth_timemask,
)

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skorch.helper import to_numpy


class AugmentedBCISolver(BaseSolver, ABC):
    """Base class for solvers that use augmented data.
    This class implements some basic methods from another methods ihnerited.
    """

    @abstractmethod
    def set_objective(self, **objective_dict):
        pass

    @property
    def name(self):
        pass

    def run(self, n_iter):
        """Run the solver to evaluate it for a given number of iterations."""
        if self.augmentation == "ChannelsDropout":
            X, y = channels_dropout(self.X, self.y, n_augmentation=n_iter)

        elif self.augmentation == "SmoothTimeMask":
            X, y = smooth_timemask(
                self.X, self.y, n_augmentation=n_iter, sfreq=self.sfreq
            )
        else:
            X = to_numpy(self.X)
            y = self.y

        self.clf.fit(X, y)

    def get_next(self, n_iter):
        return n_iter + 1

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.clf

    def warmup_solver(self):
        pass
