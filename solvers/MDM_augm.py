from benchopt import BaseSolver, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.pipeline import make_pipeline
    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM
    from benchopt.stopping_criterion import SingleRunCriterion
    from benchmark_utils import transformX_moabb
    import torch
    from braindecode.augmentation import ChannelsDropout
# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.


class Solver(BaseSolver):

    name = 'MDM_augm'

    install_cmd = 'conda'
    requirements = ['pyriemann']

    stopping_criterion = SingleRunCriterion()

    def set_objective(self, X, y, sfreq):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        seed = 20200220
        transform = ChannelsDropout(probability=0.5,
                                    p_drop=0.2,
                                    random_state=seed)

        X_tr, _ = transform.operation(torch.as_tensor(X).float(),
                                      None,
                                      mask_start_per_sample=[0 for i in range(len(X))],
                                     )
        X_tr = X_tr.numpy()
        X_transform = transformX_moabb(X)
        X_augm = X_transform + X_tr
        y_augm = y+y
        self.X, self.y = X_augm, y_augm
        self.clf = make_pipeline(Covariances("oas"),
                                 MDM(metric="riemann")
                                 )

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        self.clf.fit(self.X, self.y)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.clf

    def warmup_solver(self):
        pass
