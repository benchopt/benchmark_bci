from benchopt import safe_import_context
from benchmark_utils.augmented_dataset import AugmentedBCISolver


with safe_import_context() as import_ctx:
    from numpy import double

    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import FunctionTransformer
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    from mne.decoding import CSP

    from skorch.helper import to_numpy


class Solver(AugmentedBCISolver):
    name = "CSPLDA"
    parameters = {
        "n_components": [8],
        **AugmentedBCISolver.parameters
    }

    install_cmd = "conda"
    requirements = ["mne"]

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
            FunctionTransformer(lambda X: to_numpy(X).astype(double)),
            CSP(n_components=self.n_components),
            LDA(),
        )
