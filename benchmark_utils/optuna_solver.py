from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import optuna
    from sklearn.base import clone
    from sklearn.model_selection import cross_validate
    from sklearn.dummy import DummyClassifier
    from skorch.helper import SliceDataset, to_numpy


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class OptunaSolver(BaseSolver):

    stopping_criterion = SufficientProgressCriterion(
        strategy="callback", patience=5
    )

    def set_objective(self, X, y, sfreq):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.sfreq = sfreq
        self.X = X
        self.y = y
        if isinstance(self.y, SliceDataset):
            self.y = to_numpy(self.y)

        self.model = self.get_model()

    def objective(self, trial):

        param = self.sample_parameters(trial)
        params = {
                f"pipeline__{step_name}__{p}": v
                for step_name, step in param.items()
                for p, v in step.items()
            }

        model = clone(self.model).set_params(**params)

        cross_score = cross_validate(
            model, self.X, self.y, return_estimator=True
        )

        trial.set_user_attr("model", model.fit(self.X, self.y))

        return cross_score["test_score"].mean()

    def run(self, callback):

        # Creating a sampler to sample the hyperparameters.
        sampler = optuna.samplers.RandomSampler()
        # Initialize the best model with a dummy classifier, the optuna
        # need to beat this model.
        self.best_model = DummyClassifier().fit(self.X, self.y)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        while callback():
            study.optimize(self.objective, n_trials=10)
            self.best_model = study.best_trial.user_attrs["model"]

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(model=self.best_model)
