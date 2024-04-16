from benchopt import BaseObjective, safe_import_context


with safe_import_context() as import_ctx:

    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import FunctionTransformer

    from sklearn.metrics import balanced_accuracy_score as BAS

    from skorch.helper import to_numpy
    from benchmark_utils.splitter import IntraSessionSplitter
    from benchmark_utils.splitter import InterSessionSplitter
    from benchmark_utils.splitter import InterSubjectSplitter


class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Brain-Computer Interface"

    intall_cmd = 'conda'
    requirements = [
        'scikit-learn',
        'pytorch:pytorch',
        'pip:braindecode',
        'pip:git+https://github.com/Neurotechx/moabb@develop#egg=moabb',
    ]

    parameters = {
        'evaluation_process': [
            'intra_session',
            'inter_sessions',
            'inter_subjects',
        ],
    }

    is_convex = False

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5.2"

    def set_data(self, dataset, sfreq):
        """Set the data retrieved from Dataset.get_data.

        Data
        ----
        dataset: an instance of a braindecode.WindowsDataset
        sfreq: the sampling frequency of the data.
        """

        self.dataset = dataset
        self.sfreq = sfreq

        if self.evaluation_process == 'intra_session':
            self.cv = IntraSessionSplitter(n_folds=2)
        elif self.evaluation_process == 'inter_sessions':
            self.cv = InterSessionSplitter()
        elif self.evaluation_process == 'inter_subjects':
            self.cv = InterSubjectSplitter(n_folds=2)
        else:
            raise ValueError(
                f"unknown evaluation process '{self.evaluation_process}'"
            )

        self.cv_metadata = dict(df_meta=dataset.get_metadata())

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

        score_train = model.score(self.X_train, to_numpy(self.y_train))
        score_test = model.score(self.X_test, to_numpy(self.y_test))
        bl_acc = BAS(to_numpy(self.y_test), model.predict(self.X_test))

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
        X_train, _, y_train, _ = self.get_split(self.dataset)
        return dict(model=clf.fit(X_train, y_train))

    def split(self, cv_fold, *arrays):
        return cv_fold

    def get_objective(self):
        """Pass the objective information to Solvers.set_objective.

        Objective
        ---------
        X: training data for the model
        y: training labels to train the model.
        sfreq: sampling frequency to allow filtering the data.
        """

        self.X_train, self.X_test, self.y_train, self.y_test = (
            self.get_split(self.dataset)
        )

        return dict(
            X=self.X_train,
            y=self.y_train,
            sfreq=self.sfreq,
        )
