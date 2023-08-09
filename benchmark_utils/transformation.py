from benchopt import safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from torch import as_tensor
    from skorch.helper import to_numpy
    from braindecode.augmentation import ChannelsDropout, SmoothTimeMask


def gen_seed():
    # Iterator that generates random seeds for reproducibility reasons
    seed = 0
    while True:
        yield seed
        seed += 1


def channels_dropout(
    X, y, n_augmentation, probability=0.5, p_drop=0.2
):
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

    seed = gen_seed()
    X_augm = to_numpy(X)
    y_augm = y
    for _ in range(n_augmentation):
        transform = ChannelsDropout(
                                    probability=probability,
                                    random_state=next(seed)
                                    )
        X_tr, _ = transform.operation(
            as_tensor(X).float(), None, p_drop=p_drop
        )

        X_tr = X_tr.numpy()
        X_augm = np.concatenate((X_augm, X_tr))
        y_augm = np.concatenate((y_augm, y))

    return X_augm, y_augm


def smooth_timemask(
    X, y, n_augmentation, sfreq, probability=0.8, second=0.2
):
    """
    Function to apply smooth time mask to X raw data
    and concatenate it to the original data.
    """

    seed_generator = gen_seed()
    X_torch = as_tensor(np.array(X)).float()
    y_torch = as_tensor(y).float()
    X_augm = to_numpy(X)
    y_augm = y

    mls = int(sfreq * second)
    for _ in range(n_augmentation):
        seed = next(seed_generator)
        transform = SmoothTimeMask(
            probability=probability,
            mask_len_samples=mls,
            random_state=rng
        )

        param_augm = transform.get_augmentation_params(X_torch, y_torch)
        mls = param_augm["mask_len_samples"]
        msps = param_augm["mask_start_per_sample"]

        X_tr, _ = transform.operation(
            X_torch, None, mask_len_samples=mls, mask_start_per_sample=msps
        )
        X_tr = X_tr.numpy()
        X_augm = np.concatenate((X_augm, X_tr))
        y_augm = np.concatenate((y_augm, y))

    return X_augm, y_augm
