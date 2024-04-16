from sklearn.model_selection import ShuffleSplit
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
