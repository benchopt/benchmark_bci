from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from numpy import array

    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score as BAS

    from skorch.helper import SliceDataset, to_numpy
    from benchmark_utils.dataset import split_windows_train_test
# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.


class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "BCI"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.

    link = 'pip: git+https://github.com/Neurotechx/moabb@develop#egg=moabb'
    intall_cmd = 'conda'
    requirements = [link,
                    'scikit-learn']

    parameters = {
        'evaluation_process, subject, subject_test, session_test': [
            ('intra_subject', 1, None, None),
        ],
    }
    # The solvers will train on all the subject except subject_test.
    # It will be the same for the sessions.
    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.3.2"

    def set_data(self, dataset, paradigm_name, sfreq):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.

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

    def compute(self, model):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        if not type(model) == 'braindecode.classifier.EEGClassifier':
            self.X_train = to_numpy(self.X_train)
            self.X_test = to_numpy(self.X_test)

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
            sfreq=self.sfreq,
        )
