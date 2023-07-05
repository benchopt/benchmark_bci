from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.pipeline import make_pipeline
    from mne.decoding import CSP
    from skorch.helper import to_numpy

    from benchmark_utils.transformation import (channels_dropout,
                                                smooth_timemask)


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.


class Solver(BaseSolver):
    name = "CSPLDA"
    parameters = {
        "augmentation": [
            ("SmoothTimeMask"),
            ("ChannelsDropout"),
            ("IdentityTransform"),
        ]
    }

    install_cmd = "conda"
    requirements = ["mne", "pip:torch", "pip:braindecode"]

    def set_objective(self, X, y, sfreq):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

        self.X, self.y = X, y
        self.clf = make_pipeline(CSP(n_components=8), LDA())

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
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
