from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from skorch.helper import to_numpy
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import FunctionTransformer
    from pyriemann.estimation import Covariances

    from benchmark_utils.tWDA import tWDA


class Solver(BaseSolver):
    name = "Cov-tWDA"
    parameters = {
        "dfs": [None],
        "covariances_estimator": ["scm"],
        **BaseSolver.parameters
    }

    install_cmd = "conda"
    requirements = ["pip:pymanopt"]
    sampling_strategy = 'run_once'

    def set_objective(self, X, y, sfreq, extra_info):
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
        self.n = to_numpy(X).shape[2] - 1
        self.clf = make_pipeline(
            FunctionTransformer(to_numpy),
            Covariances(estimator=self.covariances_estimator),
            tWDA(n=self.n, dfs=self.dfs)
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
