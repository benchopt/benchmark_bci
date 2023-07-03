from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from mne.decoding import CSP
    from sklearn.pipeline import make_pipeline
    from benchopt.stopping_criterion import SingleRunCriterion
    from benchmark_utils import transformX_moabb
    from benchmark_utils import channels_dropout, smooth_timemask

# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.


class Solver(BaseSolver):
    name = "CSPLDA"
    parameters = {
        "augmentation, n_augmentation": [
            ("SmoothTimeMask", 2),
            ("ChannelsDropout", 2),
            ("ChannelsDropout", 3),
            ("ChannelsDropout", 5),
            ("SmoothTimeMask", 3),
            ("SmoothTimeMask", 5),
            ("IdentityTransform", None),
        ]
    }

    stopping_criterion = SingleRunCriterion()

    install_cmd = "conda"
    requirements = ["mne", "pip:torch", "pip:braindecode"]

    def set_objective(self, X, y, sfreq):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

        if self.augmentation == "ChannelsDropout":
            X, y = channels_dropout(X, y, n_augmentation=self.n_augmentation)

        elif self.augmentation == "SmoothTimeMask":
            X, y = smooth_timemask(
                X, y, n_augmentation=self.n_augmentation, sfreq=sfreq
            )

        else:
            X = transformX_moabb(X)

        self.X, self.y = X, y
        self.clf = make_pipeline(CSP(n_components=8), LDA())

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        self.clf.fit(self.X, self.y)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's APIafor solvers' results.
        # it is customizable for each benchmark.
        return self.clf

    def warmup_solver(self):
        pass
