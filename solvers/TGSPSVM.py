from benchopt import safe_import_context, BaseSolver


with safe_import_context() as import_ctx:
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import FunctionTransformer

    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    from skorch.helper import to_numpy


# The benchmark solvers must be named `Solver` and
# inherit from `AugmentedBCISolver` for `BCI benchmark` to work properly.
class Solver(BaseSolver):

    name = "TGSPSVM"
    parameters = {
        "covariances_estimator": ["oas"],
        "tangentspace_metric": ["riemann"],
        "svm_kernel": ["linear"],
    }

    install_cmd = "conda"
    requirements = ["pyriemann"]

    sampling_strategy = "run_once"

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

    def run(self, _):
        """Run the solver to evaluate it for a given number of augmentation.

        With this dataset, we consider that the performance curve is sampled
        for various number of augmentation applied to the dataset.
        """
        self.clf.fit(self.X, self.y)

    def get_result(self):
        """Return the model to `Objective.evaluate_result`.

        Result
        ------
        model: an instance of a fitted model.
            This model should have methods `score` and `predict`, that accept
            braindecode.WindowsDataset as input.
        """
        return dict(model=self.clf)
