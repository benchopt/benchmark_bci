from benchopt import safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC

    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    from skorch.helper import to_numpy

    from benchmark_utils.augmented_dataset import (
        AugmentedBCISolver,
    )
# The benchmark solvers must be named `Solver` and
# inherit from `AugmentedBCISolver` for `BCI benchmark` to work properly.


class Solver(AugmentedBCISolver):

    name = "TGSPSVM"
    parameters = {
        "augmentation": [
            "SmoothTimeMask",
            "ChannelsDropout",
            "IdentityTransform",
        ],
        "covariances_estimator": ["oas"],
        "tangentspace_metric": ["riemann"],
        "svm_kernel": ["linear"],
    }

    install_cmd = "conda"
    requirements = ["pyriemann"]

    def set_objective(self, X, y, sfreq):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

        self.sfreq = sfreq
        self.X = to_numpy(X)
        self.y = y
        self.clf = make_pipeline(
            Covariances(estimator=self.covariances_estimator),
            TangentSpace(metric=self.tangentspace_metric),
            SVC(kernel=self.svm_kernel),
        )
