from benchopt import safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.pipeline import make_pipeline
    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM
    from benchmark_utils.transformation import (
        channels_dropout,
        smooth_timemask,
    )
    from benchmark_utils.augmented_dataset import (
        AugmentedBCISolver,
    )
    from benchmark_utils.augmented_method import (
        Covariances_augm
    )

    from skorch.helper import to_numpy
# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.


class Solver(AugmentedBCISolver):
    name = "MDM"
    parameters = {
        "augmentation": [
            "SmoothTimeMask",
            "Barycenter"
        ]
    }

    install_cmd = "conda"
    requirements = ["pyriemann"]

    def set_objective(self, X, y, sfreq):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

        self.sfreq = X.datasets[0].raw.info["sfreq"]
        self.X = to_numpy(X)
        self.y = y
        self.clf = make_pipeline(Covariances("oas"), MDM(metric="riemann"))

    def run(self, n_iter):
        """
        Here, we override the `run` method of the `AugmentedBCISolver` class.
        to adding the Barycenter augmentation method implemented by Cristopher.
        """

        # This is the function that is called to evaluate the solver.
        if self.augmentation == "ChannelsDropout":
            X, y = channels_dropout(self.X, self.y, n_augmentation=n_iter)

        elif self.augmentation == "SmoothTimeMask":
            X, y = smooth_timemask(
                self.X, self.y, n_augmentation=n_iter, sfreq=self.sfreq
            )
            self.clf = make_pipeline(Covariances("oas"), MDM(metric="riemann"))
        elif self.augmentation == 'Barycenter':
            X = to_numpy(self.X)
            y = self.y
            self.clf = make_pipeline(Covariances_augm("cov"),
                                     MDM(metric="riemann"))
        else:
            X = to_numpy(self.X)
            y = self.y

        self.clf.fit(X, y)
