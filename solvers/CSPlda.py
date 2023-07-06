from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.pipeline import make_pipeline
    from mne.decoding import CSP
    from skorch.helper import to_numpy

    from benchmark_utils.augmented_dataset import (
        AugmentedBCISolver,
    )


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.


class Solver(AugmentedBCISolver):
    name = "CSPLDA"
    parameters = {
        "augmentation": [
            "SmoothTimeMask",
            "ChannelsDropout",
            "IdentityTransform",
        ]
    }

    install_cmd = "conda"
    requirements = ["mne"]

    def set_objective(self, X, y, sfreq):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

        self.sfreq = X.datasets[0].raw.info["sfreq"]
        self.X = to_numpy(X)
        self.y = y
        self.clf = make_pipeline(CSP(n_components=8), LDA())
