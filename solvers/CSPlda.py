from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from mne.decoding import CSP
    from sklearn.pipeline import make_pipeline
    from benchopt.stopping_criterion import SingleRunCriterion
    from benchmark_utils import transformX_moabb
    import torch
    from braindecode.augmentation import ChannelsDropout, SmoothTimeMask
    import numpy as np

# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.


class Solver(BaseSolver):

    name = 'CSPLDA'
    parameters = {'augmentation, n_augmentation': [
        ('SmoothTimeMask', 2),
        ('ChannelsDropout', 2),
        ('ChannelsDropout', 3),
        ('ChannelsDropout', 5),
        ('SmoothTimeMask', 3),
        ('SmoothTimeMask', 5),
        ('IdentityTransform', None)
                  ]}

    stopping_criterion = SingleRunCriterion()

    install_cmd = 'conda'
    requirements = ['mne', 'pip:torch', 'pip:braindecode']

    def set_objective(self, X, y, sfreq):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.

        seed = 20200220

        if self.augmentation == 'ChannelsDropout':

            transform = ChannelsDropout(probability=0.5,
                                        random_state=seed)

            X_augm = transformX_moabb(X)
            y_augm = y
            for i in range(self.n_augmentation):

                X_tr, _ = transform.operation(torch.as_tensor(X).float(),
                                              None,
                                              p_drop=0.2)

                X_tr = X_tr.numpy()
                X_augm = np.concatenate((X_augm, X_tr))
                y_augm = np.concatenate((y_augm, y))
            X = X_augm
            y = y_augm

        elif self.augmentation == 'SmoothTimeMask':
            second = 0.1

            transform = SmoothTimeMask(probability=0.5,
                                       mask_len_samples=int(sfreq * second),
                                       random_state=seed)

            X_torch = torch.as_tensor(X).float()
            y_torch = torch.as_tensor(y).float()
            param_augm = transform.get_augmentation_params(X_torch,
                                                           y_torch)
            mls = param_augm['mask_len_samples']
            msps = param_augm['mask_start_per_sample']

            X_augm = transformX_moabb(X)
            y_augm = y

            for i in range(self.n_augmentation):

                X_tr, _ = transform.operation(
                        X_torch,
                        None,
                        mask_len_samples=mls,
                        mask_start_per_sample=msps
                        )

                X_tr = X_tr.numpy()
                X_augm = np.concatenate((X_augm, X_tr))
                y_augm = np.concatenate((y_augm, y))
            X = X_augm
            y = y_augm

        else:
            X = transformX_moabb(X)

        self.X, self.y = X, y
        self.clf = make_pipeline(CSP(n_components=8), LDA())

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        self.clf.fit(self.X, self.y)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's APIafor solvers' results.
        # it is customizable for each benchmark.
        return self.clf

    def warmup_solver(self):
        pass
