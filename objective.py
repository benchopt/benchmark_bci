from benchopt import BaseObjective, safe_import_context


with safe_import_context() as import_ctx:
    from numpy import array

    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import FunctionTransformer

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score as BAS

    from skorch.helper import SliceDataset, to_numpy
    from benchmark_utils.dataset import split_windows_train_test


class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Brain-Computer Interface"

    intall_cmd = 'conda'
    requirements = [
        'scikit-learn',
        'pip:braindecode',
        'pip:git+https://github.com/Neurotechx/moabb@develop#egg=moabb',
    ]

    parameters = {
        'evaluation_process, subject, subject_test, session_test': [
            ('intra_subject', 1, None, None),
        ],
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.4.1"

    def set_data(self, dataset, sfreq):
        """Set the data retrieved from Dataset.get_data.

        Data
        ----
        Dataset: an instance of a braindecode.WindowsDataset
        sfreq: the sampling frequency of the data.
        """

        data_split_subject = dataset.split('subject')

        if self.evaluation_process == 'intra_subject':

            dataset = data_split_subject[str(self.subject)]
            X = SliceDataset(dataset, idx=0)
            y = array(list(SliceDataset(dataset, idx=1)))

            # maybe we need to do here different process for each subjects

            X_train, X_test, y_train, y_test = train_test_split(X, y)
            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test

        elif self.evaluation_process == 'inter_subject':

            sujet_test = self.subject_test
            data_subject_test = data_split_subject[str(sujet_test)]
            n_subject = len(data_split_subject)
            data_subject_train = []
            for i in range(1, n_subject+1):
                if i != sujet_test:
                    data_subject_train += data_split_subject[str(i)]

            splitted_data = split_windows_train_test(data_subject_train,
                                                     data_subject_test)

            self.X_train = splitted_data['X_train']
            self.y_train = splitted_data['y_train']
            self.X_test = splitted_data['X_test']
            self.y_test = splitted_data['y_test']

        elif self.evaluation_process == 'inter_session':

            data_subject = data_split_subject[str(self.subject)]
            data_split_session = data_subject.split('session')
            session_test = self.session_test
            data_session_test = data_split_session[session_test]
            data_session_train = []
            for key in data_split_session.items():
                if key[0] != str(session_test):
                    data_session_train += data_split_session[key[0]]

            splitted_data = split_windows_train_test(data_session_train,
                                                     data_session_test)

            self.X_train = splitted_data['X_train']
            self.y_train = splitted_data['y_train']
            self.X_test = splitted_data['X_test']
            self.y_test = splitted_data['y_test']

        self.sfreq = sfreq

        return dict(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            sfreq=sfreq,
        )

    def evaluate_result(self, model):
        """Compute the evaluation metrics for the benchmark.

        Result
        ------
        model: an instance of a fitted model.
            This model should have methods `score` and `predict`, that accept
            braindecode.WindowsDataset as input.

        Metrics
        -------
        score_test: accuracy on the testing set.
        score_train: accuracy on the training set.
        balanced_accuracy: balanced accuracy on the testing set
        value: error on the testing set.
        """

        score_train = model.score(self.X_train, self.y_train)
        score_test = model.score(self.X_test, self.y_test)
        bl_acc = BAS(self.y_test, model.predict(self.X_test))

        return dict(
            score_test=score_test,
            score_train=score_train,
            balanced_accuracy=bl_acc,
            value=1-score_test,
        )

    def get_one_result(self):
        """Return one dummy result.

        Result
        ------
        model: an instance of a fitted model.
            This model should have methods `score` and `predict`, that accept
            braindecode.WindowsDataset as input.
        """
        clf = make_pipeline(
            FunctionTransformer(to_numpy),
            DummyClassifier()
        )
        return dict(model=clf.fit(self.X_train, self.y_train))

    def get_objective(self):
        """Pass the objective information to Solvers.set_objective.

        Objective
        ---------
        X: training data for the model
        y: training labels to train the model.
        sfreq: sampling frequency to allow filtering the data.
        """

        return dict(
            X=self.X_train,
            y=self.y_train,
            sfreq=self.sfreq,
        )
