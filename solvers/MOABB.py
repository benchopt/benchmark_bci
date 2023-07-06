from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skorch.helper import to_numpy

    from benchmark_utils.pipeline import parser_pipelines
    from benchmark_utils.augmented_dataset import (
        AugmentedBCISolver,
    )


class Solver(AugmentedBCISolver):
    name = "MOABBPipelines"
    parameters = {
        "augmentation": ["SmoothTimeMask",
                         "IdentityTransform"],
        "pipeline": [
            "AUGTangSVMGrid",
            "MDM",
            "TangentSpaceSVMGrid",
            "CSPLDA",
            "FgMDM",
            "LogVarianceLDA",
            "DLCSPautoshLDA",
            "LogVarianceSVMgrid",
            "CSPSVMGrid",
            "TSElasticNetGrid",
            "TangentSpaceLR",
            "TRCSPLDA",
        ],
    }

    def set_objective(self, X, y, sfreq):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.sfreq = sfreq
        self.X = to_numpy(X)
        self.y = y

        self.clf = parser_pipelines()[self.pipeline]
