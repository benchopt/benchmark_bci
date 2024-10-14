import numpy as np

from sklearn.utils import check_random_state
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import BaseCrossValidator

from skorch.helper import SliceDataset


class IntraSessionSplitter(BaseCrossValidator):

    def __init__(self, n_folds):
        self.n_folds = n_folds

    def get_n_splits(self, dataset=None, df_meta=None):
        if df_meta is None:
            df_meta = dataset.get_metadata()
        n_sessions = len(df_meta.groupby(["subject", "session"]).first())
        return n_sessions * self.n_folds

    def split(self, dataset, df_meta=None):
        cv = ShuffleSplit(self.n_folds)
        for dataset_subject in dataset.split("subject").values():
            for dataset_session in dataset_subject.split("session").values():
                for i_train, i_test in cv.split(dataset_session):
                    X_train = SliceDataset(
                        dataset_session, idx=0, indices=i_train
                    )
                    y_train = SliceDataset(
                        dataset_session, idx=1, indices=i_train
                    )
                    X_test = SliceDataset(
                        dataset_session, idx=0, indices=i_test
                    )
                    y_test = SliceDataset(
                        dataset_session, idx=1, indices=i_test
                    )
                    yield X_train, X_test, y_train, y_test


class InterSessionSplitter(BaseCrossValidator):

    def get_n_splits(self, dataset=None, df_meta=None):
        if df_meta is None:
            df_meta = dataset.get_metadata()
        n_sessions = len(df_meta.groupby(["subject", "session"]).first())
        return n_sessions

    def split(self, dataset, df_meta=None):
        for dataset_subject in dataset.split("subject").values():
            cv = LeaveOneGroupOut()
            df_meta_session = dataset_subject.get_metadata()

            for i_train, i_test in cv.split(
                df_meta_session, groups=df_meta_session["session"]
            ):
                X_train = SliceDataset(dataset_subject, idx=0, indices=i_train)
                y_train = SliceDataset(dataset_subject, idx=1, indices=i_train)
                X_test = SliceDataset(dataset_subject, idx=0, indices=i_test)
                y_test = SliceDataset(dataset_subject, idx=1, indices=i_test)
                yield X_train, X_test, y_train, y_test


class InterSubjectSplitter(BaseCrossValidator):

    def __init__(self, n_folds):
        self.n_folds = n_folds

    def get_n_splits(self, dataset=None, df_meta=None):
        return self.n_folds

    def split(self, dataset, df_meta=None):
        cv = GroupShuffleSplit(self.n_folds)
        df_meta = dataset.get_metadata()
        for i_train, i_test in cv.split(df_meta, groups=df_meta["subject"]):
            X_train = SliceDataset(dataset, idx=0, indices=i_train)
            y_train = SliceDataset(dataset, idx=1, indices=i_train)
            X_test = SliceDataset(dataset, idx=0, indices=i_test)
            y_test = SliceDataset(dataset, idx=1, indices=i_test)
            yield X_train, X_test, y_train, y_test


class SamplerMetaSplitter(BaseCrossValidator):
    def __init__(
        self, base_splitter, fraction=None, n_splits=None, random_state=None
    ):

        self.base_splitter = base_splitter
        self.fraction = fraction
        self.n_splits = n_splits
        self.random_state = check_random_state(random_state)

        # Validate input parameters
        if self.fraction is not None and self.n_splits is not None:
            raise ValueError(
                "Specify either 'fraction' or 'n_splits', not both."
            )
        if self.fraction is None and self.n_splits is None:
            raise ValueError(
                "Either 'fraction' or 'n_splits' must be provided."
            )
        if self.fraction is not None:
            if not (0 < self.fraction <= 1):
                raise ValueError(
                    "'fraction' must be between 0 (exclusive)"
                    " and 1 (inclusive)."
                )
        if self.n_splits is not None:
            if not isinstance(self.n_splits, int) or self.n_splits <= 0:
                raise ValueError("'n_splits' must be a positive integer.")

    def get_n_splits(self, dataset=None, df_meta=None):

        total_splits = self.base_splitter.get_n_splits(dataset, df_meta)

        if self.fraction is not None:
            sampled_splits = max(
                1, int(np.floor(total_splits * self.fraction))
            )
            return min(sampled_splits, total_splits)
        else:
            return min(self.n_splits, total_splits)

    def split(self, dataset, df_meta=None):

        # Generate all possible splits
        all_splits = list(self.base_splitter.split(dataset, df_meta))

        total_splits = len(all_splits)

        if self.fraction is not None:
            n_sample = max(1, int(np.floor(total_splits * self.fraction)))
            n_sample = min(n_sample, total_splits)
        else:
            n_sample = min(self.n_splits, total_splits)

        # Sample unique split indices without replacement
        sampled_indices = self.random_state.choice(
            total_splits, size=n_sample, replace=False
        )

        # Yield only the sampled splits
        for idx in sampled_indices:
            yield all_splits[idx]
