from benchopt import safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from braindecode.augmentation import ChannelsDropout, SmoothTimeMask
    import torch
    from benchmark_utils import transformX_moabb
    from sklearn.base import BaseEstimator, TransformerMixin


def channels_dropout(X, y, n_augmentation):
    seed = 20200220
    transform = ChannelsDropout(probability=0.5, random_state=seed)
    X_augm = transformX_moabb(X)
    y_augm = y
    for i in range(n_augmentation):
        X_tr, _ = transform.operation(
            torch.as_tensor(X).float(), None, p_drop=0.2
        )

        X_tr = X_tr.numpy()
        X_augm = np.concatenate((X_augm, X_tr))
        y_augm = np.concatenate((y_augm, y))

    return (X_augm, y_augm)


def smooth_timemask(X, y, n_augmentation, sfreq):
    second = 0.1
    seed = 20200220

    transform = SmoothTimeMask(
        probability=0.5,
        mask_len_samples=int(sfreq * second),
        random_state=seed,
    )

    X_torch = torch.as_tensor(X).float()
    y_torch = torch.as_tensor(y).float()
    param_augm = transform.get_augmentation_params(X_torch, y_torch)
    mls = param_augm["mask_len_samples"]
    msps = param_augm["mask_start_per_sample"]

    X_augm = transformX_moabb(X)
    y_augm = y

    for i in range(n_augmentation):
        X_tr, _ = transform.operation(
            X_torch, None, mask_len_samples=mls, mask_start_per_sample=msps
        )

        X_tr = X_tr.numpy()
        X_augm = np.concatenate((X_augm, X_tr))
        y_augm = np.concatenate((y_augm, y))

    return (X_augm, y_augm)


class StandardScaler(BaseEstimator, TransformerMixin):
    """
    Function to get augmented covariance matrices from
    X raw data for riemaniann solvers

    """

    def __init__(self):
        """Init."""

    def fit(self, X, y):
        return self

    def transform(self, X):
        X_fin = []

        for i in np.arange(X.shape[0]):
            X_p = StandardScaler().fit_transform(X[i])
            X_fin.append(X_p)
        X_fin = np.array(X_fin)

        return X_fin
