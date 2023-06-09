from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split
    import numpy as np


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "BCI"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.

    requirements = ['pip:moabb', 'scikit-learn']
    intall_cmd = 'conda'

    parameters = {
        'evaluation_process, subject, cross_subject, cross_session': [
            ('IntraSubject', 1, None, None),
            ('IntraSubject', 2, None, None),
            ('InterSubject', None, [1, 2], None),
            ('InterSubject', None, [2, 1], None),
            ('InterSession', 1, None, [1, 2]),
            ('InterSession', 2, None, [1, 2])
        ],
    }

    # we have here an issue with the cross product, to much run are generated
    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.3.2"

    def set_data(self, dataset, paradigm):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        # The dictionary defines the keyword arguments
        # for `Objective.set_data`

        if self.evaluation_process == 'IntraSubject':

            X, y, _ = paradigm.get_data(dataset=dataset,
                                        subjects=[self.subject])

            X_train, X_test, y_train, y_test = train_test_split(X, y)
            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test

            return dict(X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test
                        )

        elif self.evaluation_process == 'InterSubject':
            X_0, y_0, _ = paradigm.get_data(dataset=dataset,
                                            subjects=[self.cross_subject[0]])

            X_1, y_1, _ = paradigm.get_data(dataset=dataset,
                                            subjects=[self.cross_subject[1]])

            X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0,
                                                                        y_0)
            X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1,
                                                                        y_1)
            self.X_train, self.y_train = X_train_0, y_train_0
            self.X_test, self.y_test = X_test_1, y_test_1

            return dict(X_train=X_train_0, y_train=y_train_0,
                        X_test=X_test_1, y_test=y_test_1
                        )

        elif self.evaluation_process == 'InterSession':
            X, y, metadata = paradigm.get_data(dataset=dataset,
                                               subjects=[self.subject])

            session_X = []
            session_y = []

            for session in np.unique(metadata.session):
                ix = metadata.session == session
                session_X.append(X[ix])
                session_y.append(y[ix])

            X_0 = session_X[self.cross_session[0]-1]
            y_0 = session_y[self.cross_session[0]-1]
            X_1 = session_X[self.cross_session[1]-1]
            y_1 = session_y[self.cross_session[1]-1]

            X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0,
                                                                        y_0)
            X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1,
                                                                        y_1)
            self.X_train, self.y_train = X_train_0, y_train_0
            self.X_test, self.y_test = X_test_1, y_test_1
            return dict(X_train=X_train_0, y_train=y_train_0,
                        X_test=X_test_1, y_test=y_test_1
                        )

    def compute(self, model):
        # The arguments of this function are the outputs of the
        # `Solver.get_result`. This defines the benchmark's API to pass
        # solvers' result. This is customizable for each benchmark.
        score_train = model.score(self.X_train, self.y_train)
        score_test = model.score(self.X_test, self.y_test)

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(value=score_test, score_train=score_train)

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
