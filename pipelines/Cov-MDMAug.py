from numpy import array, concatenate, unique
from numpy.random import choice
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.classification import MDM
from skorch.helper import to_numpy


class Covariances_augm(BaseEstimator, TransformerMixin):
    """Estimation of covariance matrix.

    Perform a simple covariance matrix estimation for each given input.

    Parameters
    ----------
    estimator : string, default=scm'
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.

    See Also
    --------

    """

    def __init__(self, estimator="cov"):
        """Init."""
        self.estimator = estimator

    def fit(self, X, y):
        """Fit.

        Do nothing. For compatibility purpose.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : Covariances instance
            The Covariances instance.
        """
        self.y = y
        return self

    def transform(self, X):
        """Estimate covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series.

        Returns
        -------
        covmats : ndarray, shape (n_matrices, n_channels, n_channels)
            Covariance matrices.
        """
        covmats_augm = cov_augm(X, self.y, estimator=self.estimator)
        return covmats_augm


def split_classes(X, y):
    n_classes = len(unique(y))
    liste_classe = [[] for _ in range(n_classes)]

    for i in range(len(X)):
        liste_classe[y[i] - 1].append(X[i])

    return liste_classe


def cov_augm(X, y, estimator="cov"):
    list_classe = split_classes(X, y)
    X = to_numpy(X)
    X = covariances(X, estimator=estimator)
    X_augm = []
    y_augm = []
    for i in range(len(list_classe)):
        len_classe = len(list_classe[i])
        list_index = [j for j in range(len_classe)]
        for _ in range(len_classe):
            X_rand = []
            y_augm_class = []
            list_index_rand = choice(list_index, 5)
            for index in list_index_rand:
                X_rand.append(X[index])
            X_rand = array(X_rand)
            M = mean_covariance(X_rand)
            X_augm.append(M)
            y_augm_class.append(i)

    X_augm = array(X_augm)
    y_augm = array(y_augm)
    X = concatenate((X, X_augm))
    y = concatenate((y, y_augm))
    return X, y


pipe = make_pipeline(
    Covariances_augm("oas"), MDM(metric="riemann")
)

# this is what will be loaded
PIPELINE = {
    "name": "Cov-MDMAug",
    "paradigms": ["LeftRightImagery"],
    "pipeline": pipe,
}
