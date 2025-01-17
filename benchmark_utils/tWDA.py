import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import pinvh

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax
from pyriemann.utils.base import logm
from benchmark_utils.tWishart import t_wish_est, kurtosis_estimation
from benchmark_utils.tWishart import log_generator_density


class tWDA(ClassifierMixin, BaseEstimator):
    """Bayesian classification by t-Wishart.

    Parameters
    ----------
    n : int
        number of time samples.
    dfs : list, default=None
        degree(s) of freedom of the t- modeling (shape parameters) for
        different classes.
        If None, they are estimated with the kurtosis estimation method
    n_jobs: int, default=1
        Number of jobs to run in parallel.
    rmt: bool, default=False
        if True, the RMT approximation is used whe dfs are estimated and
        not provided.

    Attributes
    ----------
    Nc : int
        Number of classes
    n : int
        number of time samples
    classes_ : ndarray, shape (Nc,)
        Labels for each class.
    centers : list of ``Nc`` ndarrays of shape (n_channels, n_channels)
        Centroids for each class.
    dfs : list of ``Nc`` floats or 'inf'
        Degrees of freedom (shape parameters) for different classes
    pi : array, shape(Nc,)
        Proportions for each class.
    """

    def __init__(self, n, dfs=None, n_jobs=1, rmt=False):
        """Init.
        """
        self.n = n
        self.dfs = dfs
        self.n_jobs = n_jobs
        self.rmt = rmt
        if not (self.dfs is None):
            assert type(self.dfs) is list, "`dfs` must be a list or None"
            assert len(self.dfs) > 0, "Empty list for `dfs` "
            for i in range(len(self.dfs)):
                if self.dfs[i] == 'inf':
                    self.dfs[i] = np.inf

    def estimate_df(self, S):
        """
        Estimates the degree of freedom (shape parameter) of the t- Wishart
        distribution with unknown center matrix and based on the provided
        samples

        Parameters
        ----------
        S : ndarray, shape (n_trials,n_channels,n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        float
            Estimated degree of freedom of the t- Wishart distribution.

        """
        return kurtosis_estimation(S, self.n, np.mean(S, axis=0)/self.n,
                                   rmt=self.rmt)

    def compute_class_center(self, S, df):
        """
        Compute the Maximum Likelihood Estimator (MLE) for the center matrix
        of the t- Wishart distribution with knowm parameters n and df, based
        on provided samples

        Parameters
        ----------
        S : ndarray, shape (n_trials,n_channels,n_channels)
            ndarray of SPD matrices.
        df : float
            Degree of freedom of the t- Wishart distribution.

        Returns
        -------
        array, shape (n_channels,n_channels)
            MLE for the center marix of the t- Wishart distribution.

        """
        if df == np.inf:
            return np.mean(S, axis=0)/self.n
        return t_wish_est(S, self.n, df=df)

    def fit(self, S, y):
        """Fit (estimates) the centroids.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        Returns
        -------
        self : tWDA classifier instance
        """
        self.classes_ = np.unique(y)
        self.Nc = len(self.classes_)
        y = np.asarray(y)
        p, _ = S[0].shape

        # estimate dfs if needed
        if self.dfs is None:
            if self.n_jobs == 1:
                self.dfs = [self.estimate_df(S[y == self.classes_[i]])
                            for i in range(self.Nc)]
            else:
                self.dfs = Parallel(n_jobs=self.n_jobs)
                (delayed(self.estimate_df)(S[y == self.classes_[i]]) for i in
                 range(self.Nc))
        else:
            if len(self.dfs) == 1:
                self.dfs = [self.dfs[0] for _ in range(self.Nc)]

        # estimate centers
        if self.n_jobs == 1:
            self.centers = [self.compute_class_center(S[y == self.classes_[i]],
                            self.dfs[i]) for i in range(self.Nc)]
        else:
            self.centers = Parallel(n_jobs=self.n_jobs)
            (delayed(self.compute_class_center)
             (S[y == self.classes_[i]], self.dfs[i]) for i in range(self.Nc))

        # estimate proportions
        self.pi = np.ones(self.Nc)
        for k in range(self.Nc):
            self.pi[k] = len(y[y == self.classes_[k]])/len(y)

        return self

    def _predict_discimination(self, covtest):
        """
        Helper to predict the discimination. equivalent to transform.

        Parameters
        ----------
        covtest : array, shape (n_channels, n_channels)
            Testing SPD matrix.

        Returns
        -------
        discrimination : array, shape (n_trials,n_classes)
            discrimination table between the testing SPD matrix and the
            centroids.

        """
        K, p, _ = covtest.shape
        discrimination = np.zeros((K, self.Nc))
        traces = np.zeros((K, self.Nc))

        if len(np.unique(np.asarray(self.dfs))) == 1:
            # if a common df is used for all the classes:
            log_h = log_generator_density(self.n, p, self.dfs[0],
                                          neglect_df_terms=True)

        for i in range(self.Nc):
            if len(np.unique(np.asarray(self.dfs))) != 1:
                # if different dfs are used for all classes
                log_h = log_generator_density(self.n, p, self.dfs[i])

            center = self.centers[i].copy()
            inv_center = pinvh(center)
            logdet_center = np.trace(logm(center))
            for j in range(K):
                # discrimination between the center of class i and the cov_j
                trace = np.matrix.trace(inv_center @ covtest[j])
                traces[j, i] = trace
                discrimination[j, i] = np.log(self.pi[i]) -\
                    0.5*self.n*logdet_center+log_h(trace)

        return discrimination

    def predict(self, covtest):
        """get the predictions.
        Parameters
        ----------
        covtest : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        preds : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        discrimination = self._predict_discimination(covtest)

        preds = []
        n_trials, n_classes = discrimination.shape
        for i in range(n_trials):
            preds.append(self.classes_[discrimination[i, :].argmax()])
        preds = np.asarray(preds)
        return preds

    def fit_predict(self, S, y):
        """Fit and predict in one function."""
        self.fit(S, y)
        return self.predict(S)

    def predict_proba(self, S):
        """Predict proba using softmax.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        return softmax(self._predict_discrimination(S))
