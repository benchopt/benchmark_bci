from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils import OptunaSolver
    from skorch.helper import SliceDataset, to_numpy
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import FunctionTransformer

    from benchmark_utils.pipeline import parser_pipelines


class Solver(OptunaSolver):
    name = "MOABBPipelinesOptuna"
    parameters = {
        "pipeline": [
            "DUMMY",
        ],
    }

    def set_objective(self, X, y, sfreq):
        """Set the objective information from Objective.get_objective.

        Objective
        ---------
        X: training data for the model
        y: training labels to train the model.
        sfreq: sampling frequency to allow filtering the data.
        """
        self.sfreq = sfreq
        self.X = X
        self.y = y

        self.clf = make_pipeline(
            FunctionTransformer(to_numpy),
            parser_pipelines()[self.pipeline]
        )

    def get_model(self):
        return self.clf

    def get_result(self):
        """Return the model to `Objective.evaluate_result`.

        Result
        ------
        model: an instance of a fitted model.
            This model should have methods `score` and `predict`, that accept
            braindecode.WindowsDataset as input.
        """
        return dict(model=self.clf)

    def get_model(self):
        return self.clf

    def sample_parameters(self, trial):
        strategy_options = ["most_frequent", "prior", "stratified",
                            "uniform", "constant"]
        strategy_parameters = trial.suggest_categorical('strategy', strategy_options)

        return dict(
            strategy_parameters=strategy_parameters,
        )