from benchopt import safe_import_context
from benchmark_utils.augmented_dataset import AugmentedBCISolver


with safe_import_context() as import_ctx:
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import FunctionTransformer

    from pyriemann.estimation import Covariances
    from pyriemann.classification import KNearestNeighbor

    from skorch.helper import to_numpy


class Solver(AugmentedBCISolver):
    name = "K Nearest-Neighbors"

    install_cmd = "conda"
    requirements = ["pyriemann"]
    parameters = {
        "covariances_estimator": ["oas"],
        "KNN_cov_metric": ["euclid"],
        "n_neighbors": [7],
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
            Covariances(estimator=self.covariances_estimator),
            KNearestNeighbor(n_neighbors=self.n_neighbors,
                             metric=self.KNN_cov_metric),
        )
