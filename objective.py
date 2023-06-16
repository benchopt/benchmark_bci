from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score as BAS
    import numpy as np
    from skorch.helper import SliceDataset
    from benchmark_utils import transformX_moabb

# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.


class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "BCI"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.

    intall_cmd = 'conda'
    requirements = ['pip:moabb', 'scikit-learn']

    parameters = {
        'evaluation_process, subject, subject_test, session_test': [
            ('inter_session', 1, None, 'session_T'),
            ('inter_session', 2, None, 'session_E'),
            ('intra_subject', 1, None, None),
            ('intra_subject', 2, None, None),
            ('inter_subject', None, 3, None),

        ],
    }
    # The solvers will train on all the subject except subject_test.
    # It will be the same for the sessions.
    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.3.2"

    def set_data(self, dataset, paradigm_name):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        # The dictionary defines the keyword arguments
        # for `Objective.set_data`

        n_channels = dataset[0][0].shape[0]
        input_window_samples = dataset[0][0].shape[1]
        data_split_subject = dataset.split('subject')
        self.n_channels = n_channels
        self.input_window_samples = input_window_samples

        if self.evaluation_process == 'intra_subject':

            dataset = data_split_subject[str(self.subject)]
            X = SliceDataset(dataset, idx=0)
            y = np.array([y for y in SliceDataset(dataset, idx=1)])

            # maybe we need to do here differentt process for each subjects

            X_train, X_test, y_train, y_test = train_test_split(X, y)
            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test

        elif self.evaluation_process == 'inter_subject':

            n_channels = dataset[0][0].shape[0]
            input_window_samples = dataset[0][0].shape[1]
            sujet_test = self.subject_test
            data_subject_test = data_split_subject[str(sujet_test)]
            n_subject = len(data_split_subject)
            data_subject_train = []
            for i in range(1, n_subject+1):
                if i != sujet_test:
                    data_subject_train += data_split_subject[str(i)]

            X_test = SliceDataset(data_subject_test, idx=0)
            y_test = np.array([y for y in SliceDataset(data_subject_test,
                                                       idx=1)])

            X_train = SliceDataset(data_subject_train, idx=0)
            y_train = np.array([y for y in SliceDataset(data_subject_train,
                                                        idx=1)])

            self.X_train, self.y_train = X_train,  y_train
            self.X_test, self.y_test = X_test, y_test

        elif self.evaluation_process == 'inter_session':

            n_channels = dataset[0][0].shape[0]
            input_window_samples = dataset[0][0].shape[1]
            data_subject = data_split_subject[str(self.subject)]
            data_split_session = data_subject.split('session')
            session_test = self.session_test
            data_session_test = data_split_session[session_test]
            data_session_train = []
            for clé in data_split_session.items():
                if clé != session_test:
                    data_session_train += data_split_session[clé[0]]
            X_test = SliceDataset(data_session_test, idx=0)
            y_test = np.array([y for y in SliceDataset(data_session_test,
                                                       idx=1)])
            X_train = SliceDataset(data_session_train, idx=0)
            y_train = np.array([y for y in SliceDataset(data_session_train,
                                                        idx=1)])

            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test

        return dict(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                n_channels=n_channels,
                input_window_samples=input_window_samples
            )

    def compute(self, model):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        if not type(model) == 'braindecode.classifier.EEGClassifier':
            self.X_train = transformX_moabb(self.X_train)
            self.X_test = transformX_moabb(self.X_test)

        score_train = model.score(self.X_train, self.y_train)
        score_test = model.score(self.X_test, self.y_test)
        bl_acc = BAS(self.y_test, model.predict(self.X_test))

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(score_test=score_test,
                    value=-score_test,
                    score_train=score_train,
                    balanced_accuracy=bl_acc)

    def get_one_solution(self):
        # Return one solution. The return value should be an object compatible
        # with `self.compute`. This is mainly for testing purposes.
        return DummyClassifier().fit(self.X_train, self.y_train)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.

        return dict(
            X=self.X_train,
            y=self.y_train,
            n_channels=self.n_channels,
            input_window_samples=self.input_window_samples,
        )
