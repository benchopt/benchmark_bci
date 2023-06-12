from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    import numpy as np
    from benchmark_utils import flatten, list_train_test
    from sklearn.metrics import balanced_accuracy_score as BAS


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
            ('intra_subject', 1, None, None),
            ('intra_subject', 2, None, None),
            ('inter_subject', None, 3, None),
            ('inter_session', 1, None, 2),
            ('inter_session', 2, None, 1)
        ],
    }

    # The solvers will train on all the subject except subject_test.
    # It will be the same for the sessions.
    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.3.2"

    def set_data(self, dataset, paradigm):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        # The dictionary defines the keyword arguments
        # for `Objective.set_data`

        if self.evaluation_process == 'intra_subject':

            X, y, _ = paradigm.get_data(dataset=dataset,
                                        subjects=[self.subject])

            X_train, X_test, y_train, y_test = train_test_split(X, y)
            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test

            return dict(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test
            )

        elif self.evaluation_process == 'inter_session':
            X, y, metadata = paradigm.get_data(dataset=dataset,
                                               subjects=[self.subject])

            session_X = []
            session_y = []

            for session in np.unique(metadata.session):
                ix = metadata.session == session
                session_X.append(X[ix])
                session_y.append(y[ix])

            X_test = session_X[self.session_test-1]
            _, X_train = list_train_test(self.session_test-1, session_X)
            X_train = np.array(flatten(X_train))
            y_test = session_y[self.session_test-1]
            _, y_train = list_train_test(self.session_test-1, session_y)
            y_train = np.array(flatten(y_train))

            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test
            return dict(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test
            )

        elif self.evaluation_process == 'inter_subject':

            subjects = dataset.subject_list
            subject_test, subjects_train = list_train_test(self.subject_test-1,
                                                           subjects)
            X_train, y_train, _ = paradigm.get_data(dataset=dataset,
                                                    subjects=subjects_train)

            X_test, y_test, _ = paradigm.get_data(dataset=dataset,
                                                  subjects=subject_test)

            self.X_train, self.y_train = X_train,  y_train
            self.X_test, self.y_test = X_test, y_test

            return dict(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test
            )

    def compute(self, model):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
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
        )
