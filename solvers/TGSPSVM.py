from benchopt import safe_import_context
from benchmark_utils.augmented_dataset import AugmentedBCISolver


with safe_import_context() as import_ctx:
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import FunctionTransformer

    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    from skorch.helper import to_numpy


# The benchmark solvers must be named `Solver` and
# inherit from `AugmentedBCISolver` for `BCI benchmark` to work properly.
class Solver(AugmentedBCISolver):

    name = "TGSPSVM"
    parameters = {
        "augmentation": [
            "SmoothTimeMask",
        ],
        "covariances_estimator": ["oas"],
        "tangentspace_metric": ["riemann"],
        "svm_kernel": ["linear"],
        **AugmentedBCISolver.parameters,
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
            TangentSpace(metric=self.tangentspace_metric),
            SVC(kernel=self.svm_kernel),
        )
