from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    from sklearn.dummy import DummyClassifier


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.


class Solver(BaseSolver):

    name = 'Dummyclassifier'

    stopping_criterion = SingleRunCriterion()

    def set_objective(self, X, y):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.X, self.y = X, y
        self.clf = DummyClassifier()

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        self.clf.fit(self.X, self.y)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.clf

    def warmup_solver(self):
        pass
