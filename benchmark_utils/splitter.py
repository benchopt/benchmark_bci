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
        n_sessions = len(df_meta.groupby(['subject', 'session']).first())
        return n_sessions * self.n_folds

    def split(self, dataset, df_meta=None):
        cv = ShuffleSplit(self.n_folds)
        for dataset_subject in dataset.split('subject').values():
            for dataset_session in dataset_subject.split('session').values():
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
        n_sessions = len(df_meta.groupby(['subject', 'session']).first())
        return n_sessions

    def split(self, dataset, df_meta=None):
        for dataset_subject in dataset.split('subject').values():
            cv = LeaveOneGroupOut()
            df_meta_session = dataset_subject.get_metadata()

            for i_train, i_test in cv.split(df_meta_session,
                                            groups=df_meta_session['session']):
                X_train = SliceDataset(
                    dataset_subject, idx=0, indices=i_train
                )
                y_train = SliceDataset(
                    dataset_subject, idx=1, indices=i_train
                )
                X_test = SliceDataset(
                    dataset_subject, idx=0, indices=i_test
                )
                y_test = SliceDataset(
                    dataset_subject, idx=1, indices=i_test
                )
                yield X_train, X_test, y_train, y_test


class InterSubjectSplitter(BaseCrossValidator):

    def __init__(self, n_folds):
        self.n_folds = n_folds

    def get_n_splits(self, dataset=None, df_meta=None):
        return self.n_folds

    def split(self, dataset, df_meta=None):
        cv = GroupShuffleSplit(self.n_folds)
        df_meta = dataset.get_metadata()
        for i_train, i_test in cv.split(df_meta, groups=df_meta['subject']):
            X_train = SliceDataset(
                dataset, idx=0, indices=i_train
            )
            y_train = SliceDataset(
                dataset, idx=1, indices=i_train
            )
            X_test = SliceDataset(
                dataset, idx=0, indices=i_test
            )
            y_test = SliceDataset(
                dataset, idx=1, indices=i_test
            )
            yield X_train, X_test, y_train, y_test
