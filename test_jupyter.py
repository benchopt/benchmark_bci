#%%
from benchopt import safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from numpy import concatenate
    from torch import as_tensor
    from skorch.helper import to_numpy
    from braindecode.augmentation import ChannelsDropout, SmoothTimeMask
    from pyriemann.utils.covariance import covariances
    from pyriemann.utils.mean import mean_covariance
    from sklearn.base import BaseEstimator, TransformerMixin
    from braindecode.datasets import MOABBDataset

from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from numpy import multiply, array
    from skorch.helper import SliceDataset
    from braindecode.preprocessing import (
        preprocess,
        Preprocessor,
    )
    from braindecode.preprocessing import create_windows_from_events


def pre_process_windows_dataset(
    dataset, low_cut_hz=4.0, high_cut_hz=38.0, factor=1e6, n_jobs=-1
):
    """
    Preprocess the window dataset.
        Function to apply preprocessing to the window (epoched) dataset.
        We proceed as follows:
        - Pick only EEG channels
        - Convert from V to uV
        - Bandpass filter
        - Apply exponential moving standardization
    Parameters:
    -----------
    dataset: WindowsDataset or BaseConcatDataset
        Dataset to use.
    low_cut_hz: float
        Low cut frequency for the bandpass filter.
    high_cut_hz: float
        High cut frequency for the bandpass filter.
    factor: float
        Factor to convert from V to uV.
    n_jobs: int
        Number of jobs to use for parallelization.
    Returns:
    --------
    dataset: WindowsDataset or BaseConcatDataset
        Preprocessed dataset.
    """
    # Parameters for exponential moving standardization
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        # Keep EEG sensors
        Preprocessor(
            lambda data, factor: multiply(data, factor),
            # Convert from V to uV
            factor=factor,
        ),
        # Bandpass filter
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
    ]

    # Transform the data
    preprocess(dataset, preprocessors, n_jobs=n_jobs)

    return dataset


def windows_data(
    dataset,
    paradigm_name,
    trial_start_offset_seconds=-0.5,
    low_cut_hz=4.0,
    high_cut_hz=38.0,
    factor=1e6,
    n_jobs=-1,
):
    """Create windows from the dataset.

    Parameters:
    -----------
    dataset: MOABBDataset
        Dataset to use.
    paradigm_name: str
        Name of the paradigm to use.
    Returns:
    --------
    windows_dataset: WindowsDataset
        Dataset with windows.
    sfreq: float
        Sampling frequency of the dataset.
    """
    # Define mapping of classes to integers
    # We use two classes from the dataset
    # 1. left-hand vs right-hand motor imagery
    if paradigm_name == "LeftRightImagery":
        mapping = {"left_hand": 1, "right_hand": 2}

    elif paradigm_name == "MotorImagery":
        mapping = {"left_hand": 1, "right_hand": 2, "feet": 4, "tongue": 3}

    dataset = pre_process_windows_dataset(
        dataset,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        factor=factor,
        n_jobs=n_jobs,
    )

    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    # Create windows using braindecode function for this.
    # It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
        mapping=mapping,
    )

    return windows_dataset, sfreq


def split_windows_train_test(data_subject_test, data_subject_train):
    """
    Split the window dataset into train and test sets.
    """
    # Converting the windows dataset into numpy arrays
    X_test = SliceDataset(data_subject_test, idx=0)
    y_test = array(list(SliceDataset(data_subject_test, idx=1)))

    X_train = SliceDataset(data_subject_train, idx=0)
    y_train = array(list(SliceDataset(data_subject_train, idx=1)))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


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

    X_augm = to_numpy(X)
    y_augm = y
    for i in range(n_augmentation):
        transform = ChannelsDropout(probability=probability)
        X_tr, _ = transform.operation(
            as_tensor(X).float(), None, p_drop=p_drop
        )

        X_tr = X_tr.numpy()
        X_augm = concatenate((X_augm, X_tr))
        y_augm = concatenate((y_augm, y))

    return X_augm, y_augm


def smooth_timemask(
    X, y, n_augmentation, sfreq, probability=0.8, second=0.2
):
    """
    Function to apply smooth time mask to X raw data
    and concatenate it to the original data.
    """

    X_torch = as_tensor(np.array(X)).float()
    y_torch = as_tensor(y).float()
    X_augm = to_numpy(X)
    y_augm = y

    mask_len_samples = int(sfreq * second)
    for i in range(n_augmentation):

        transform = SmoothTimeMask(
                                probability=probability,
                                mask_len_samples=mask_len_samples,
                                    )

        param_augm = transform.get_augmentation_params(X_torch, y_torch)
        mls = param_augm["mask_len_samples"]
        msps = param_augm["mask_start_per_sample"]

        X_tr, _ = transform.operation(
            X_torch, None, mask_len_samples=mls, mask_start_per_sample=msps
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
        covmats_augm = self.cov_augm(X, self.y, estimator=self.estimator)
        print(" les matrices augmentées sont", covmats_augm)
        print("elles sont de taille", covmats_augm.shape)
        return covmats_augm

    def split_classes(self, X, y):
        n_classes = len(np.unique(y))
        liste_classe = [[] for i in range(n_classes)]

        for i in range(len(X)):
            liste_classe[y[i]-1].append(X[i])

        return liste_classe

    def cov_augm(self, X, y, estimator='cov'):

        X = to_numpy(X)
        X = covariances(X, estimator=estimator)
        list_classe = self.split_classes(X, y)
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


dataset_name = "BNCI2014001"
data = MOABBDataset(dataset_name=dataset_name,
                    subject_ids=None)

dataset, sfreq = windows_data(data, "MotorImagery")
data_split_subject = dataset.split('subject')
dataset = data_split_subject[str(1)]
X = SliceDataset(dataset, idx=0)
y = array(list(SliceDataset(dataset, idx=1)))

# %%
