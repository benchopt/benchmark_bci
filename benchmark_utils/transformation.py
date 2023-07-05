from benchopt import safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    from numpy import concatenate
    from torch import as_tensor
    from sklearn.base import BaseEstimator, TransformerMixin
    from skorch.helper import to_numpy
    from braindecode.augmentation import ChannelsDropout, SmoothTimeMask
    from pyriemann.utils.covariance import covariances
    from pyriemann.utils.mean import mean_covariance


def channels_dropout(X, y, n_augmentation,
                     seed=0, probability=0.5, p_drop=0.2):
    """
    Function to apply channels dropout to X raw data
    and concatenate it to the original data.

    ----------
    X : array-like of shape (n_trials, n_channels, n_times)
        The input trials.
    y : array-like of shape (n_trials,)
        The labels.
    n_augmentation : int
        Number of augmentation to apply and increase the size of the dataset.
    seed : int
        Random seed.
    probability : float
        Probability of applying the tranformation.
    p_drop : float
        Probability of dropout a channel.

    Returns
    -------
    X_augm : array-like of shape (n_trials * n_augmentation,
              n_channels, n_times)
        The augmented trials.
    y_augm : array-like of shape (n_trials * n_augmentation,)
        The labels.

    """
    transform = ChannelsDropout(probability=probability,
                                random_state=seed)
    X_augm = to_numpy(X)
    y_augm = y
    for i in range(n_augmentation):
        X_tr, _ = transform.operation(
            as_tensor(X).float(), None, p_drop=p_drop
        )

        X_tr = X_tr.numpy()
        X_augm = concatenate((X_augm, X_tr))
        y_augm = concatenate((y_augm, y))

    return X_augm, y_augm


def smooth_timemask(X, y, n_augmentation, sfreq, seed=0,
                    probability=0.5, second=0.1):
    """
    Function to apply smooth time mask to X raw data
    and concatenate it to the original data.
    """

    transform = SmoothTimeMask(
        probability=probability,
        mask_len_samples=int(sfreq * second),
        random_state=seed,
    )

    X_torch = as_tensor(np.array(X)).float()
    y_torch = as_tensor(y).float()
    param_augm = transform.get_augmentation_params(X_torch, y_torch)
    mls = param_augm["mask_len_samples"]
    msps = param_augm["mask_start_per_sample"]

    X_augm = to_numpy(X)
    y_augm = y

    for i in range(n_augmentation):
        X_tr, _ = transform.operation(
            X_torch, None, mask_len_samples=mls,
            mask_start_per_sample=msps
        )

        X_tr = X_tr.numpy()
        X_augm = concatenate((X_augm, X_tr))
        y_augm = concatenate((y_augm, y))

    return X_augm, y_augm


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
    ERPCovariances
    XdawnCovariances
    CospCovariances
    HankelCovariances
    """
    install_cmd = "conda"
    requirements = ["scikit-learn"]

    def __init__(self, estimator='cov'):
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
    n_classes = len(np.unique(y))
    liste_classe = [[] for i in range(n_classes)]

    for i in range(len(X)):
        liste_classe[y[i]-1].append(X[i])

    return liste_classe


def cov_augm(X, y, estimator='cov'):
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
            list_index_rand = np.random.choice(list_index, 5)
            for index in list_index_rand:
                X_rand.append(X[index])
            X_rand = np.array(X_rand)
            M = mean_covariance(X_rand)
            X_augm.append(M)
            y_augm_class.append(i)

    X_augm = np.array(X_augm)
    y_augm = np.array(y_augm)
    X = np.concatenate((X, X_augm))
    y = np.concatenate((y, y_augm))
    return X, y
