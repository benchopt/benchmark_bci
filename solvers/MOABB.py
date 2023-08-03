from benchopt import safe_import_context
from benchmark_utils.augmented_dataset import AugmentedBCISolver


with safe_import_context() as import_ctx:
    from skorch.helper import to_numpy
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import FunctionTransformer

    from benchmark_utils.pipeline import parser_pipelines


class Solver(AugmentedBCISolver):
    name = "MOABBPipelines"
    parameters = {
        "pipeline": [
            "AUGTangSVMGrid",
            "MDM",
            "MDMAug",
            "TangentSpaceSVMGrid",
            "COVCSPLDA",
            "FgMDM",
            "LogVarianceLDA",
            "DLCSPautoshLDA",
            "LogVarianceSVMgrid",
            "COVCSPSVMGrid",
            "TSElasticNetGrid",
            "TangentSpaceLR",
            "TRCSPLDA",
            "DUMMY",
        ],
        **AugmentedBCISolver.parameters
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
