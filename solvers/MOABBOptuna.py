from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils import OptunaSolver
    from skorch.helper import SliceDataset, to_numpy
    from sklearn.pipeline import make_pipeline, FunctionTransformer

    from benchmark_utils.pipeline import (
        parser_pipelines,
        get_hyperparams_from_pipeline,
    )


class Solver(OptunaSolver):
    name = "MOABBPipelinesOptuna"
    parameters = {
        "pipeline": [
            "TRCSPLDA",
            "MDM",
            "TangentSpaceSVMGrid",
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
        if isinstance(self.y, SliceDataset):
            self.y = to_numpy(self.y)

        self.clf = make_pipeline(
            FunctionTransformer(to_numpy),
            parser_pipelines()[self.pipeline]
        )

    def get_model(self):
        return self.clf

    def sample_parameters(self, trial):
        return get_hyperparams_from_pipeline(self.pipeline, trial)
