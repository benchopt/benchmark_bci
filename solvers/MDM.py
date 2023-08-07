from benchopt import safe_import_context
from benchmark_utils.augmented_dataset import AugmentedBCISolver


with safe_import_context() as import_ctx:
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import FunctionTransformer

    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM

    from skorch.helper import to_numpy


class Solver(AugmentedBCISolver):
    name = "MDM"
    parameters = {
        "covariances_estimator": ["oas"],
        "MDM_metric": ["riemann"],
        **AugmentedBCISolver.parameters
    }

    install_cmd = "conda"
    requirements = ["pyriemann"]

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
            Covariances(estimator=self.covariances_estimator),
            MDM(metric=self.MDM_metric)
        )
