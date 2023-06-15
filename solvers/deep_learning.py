from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    from braindecode import EEGClassifier
    from skorch.callbacks import LRScheduler
    import torch
    from braindecode.models import ShallowFBCSPNet


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.

class Solver(BaseSolver):

    name = 'deep_learning'

    stopping_criterion = SingleRunCriterion()

    # here maybe we need to define for each solvers a other input named
    # metadata which would be a dictionarry where we could get n_channels
    # and input_window_samples

    def set_objective(self, X, y, n_channels, input_window_samples):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        lr = 0.0625 * 0.01
        weight_decay = 0
        batch_size = 64
        n_epochs = 4
        n_classes = 4
        print("les valeurs: ", n_channels, n_classes, input_window_samples)
        model = ShallowFBCSPNet(n_channels,
                                n_classes,
                                input_window_samples=input_window_samples,
                                final_conv_length="auto",)

        cuda = torch.cuda.is_available()
        # check if GPU is available, if True chooses to use it
        device = "cuda" if cuda else "cpu"
        if cuda:
            torch.backends.cudnn.benchmark = True

        clf = EEGClassifier(
            model,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            train_split=None,
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            callbacks=[
                "accuracy",
                ("lr_scheduler", LRScheduler("CosineAnnealingLR",
                                             T_max=n_epochs - 1)),
            ],
            device=device,
        )
        self.clf = clf

        self.X, self.y = X, y

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver
        # .
        self.clf.fit(self.X, y=self.y, epochs=40)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.clf

    def warmup_solver(self):
        pass
