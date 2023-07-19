from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    from benchopt.stopping_criterion import SingleRunCriterion
    from braindecode import EEGClassifier
    from braindecode.augmentation import (
        AugmentedDataLoader,
        ChannelsDropout,
        FTSurrogate,
        IdentityTransform,
        SmoothTimeMask,
    )
    from braindecode.models import ShallowFBCSPNet
    from numpy import linspace
    from skorch.callbacks import LRScheduler


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.


class Solver(BaseSolver):
    name = "ShallowFBCSPNet"
    parameters = {
        "augmentation": (
            "ChannelsDropout",
            "SmoothTimeMask",
            "IdentityTransform",
            "FTSurrogate",
        ),
        "lr": [0.0625 * 0.01],
        "weight_decay": [0],
        "batch_size": [64],
        "n_epochs": [1],
        "proba": [0.5],

    }

    stopping_criterion = SingleRunCriterion()

    install_cmd = "conda"
    requirements = ["pip:torch", "pip:braindecode"]

    # here maybe we need to define for each solvers a other input named
    # metadata which would be a dictionarry where we could get n_channels
    # and input_window_samples

    def set_objective(self, X, y, sfreq):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        # here we want to define à function that get the data X,y from Moabb
        # and convert it to data accessible for deep learning methodes
        lr = self.lr
        weight_decay = self.weight_decay
        batch_size = self.batch_size
        n_epochs = self.n_epochs
        n_classes = len(set(y))
        n_channels = X[0].shape[0]
        input_window_samples = X[0].shape[1]
        model = ShallowFBCSPNet(
            n_channels,
            n_classes,
            input_window_samples=input_window_samples,
            final_conv_length="auto",
        )

        cuda = torch.cuda.is_available()
        # check if GPU is available, if True chooses to use it
        device = "cuda" if cuda else "cpu"
        if cuda:
            torch.backends.cudnn.benchmark = True

        # we need to get the folowing parameter either
        # with the inputs of set object or 'manually'

        seed = 20200220

        if self.augmentation == "SmoothTimeMask":
            transforms = [
                SmoothTimeMask(
                    probability=self.proba,
                    mask_len_samples=int(sfreq * second),
                    random_state=seed,
                )
                for second in linspace(0.1, 2, 3)
            ]

        elif self.augmentation == "ChannelDropout":
            transforms = [
                ChannelsDropout(
                    probability=self.proba, p_drop=prob, random_state=seed
                )
                for prob in linspace(0, 1, 3)
            ]

        elif self.augmentation == "FTSurrogate":
            transforms = [
                FTSurrogate(
                    probability=self.proba,
                    phase_noise_magnitude=phase_freq,
                    random_state=seed,
                )
                for phase_freq in linspace(0, 1, 3)
            ]
        else:
            transforms = [IdentityTransform()]

        clf = EEGClassifier(
            model,
            iterator_train=AugmentedDataLoader,
            iterator_train__transforms=transforms,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            train_split=None,
            optimizer__lr=lr,
            max_epochs=n_epochs,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            callbacks=[
                "accuracy",
                (
                    "lr_scheduler",
                    LRScheduler("CosineAnnealingLR",
                                T_max=n_epochs - 1),
                ),
            ],
            device=device,
        )

        self.clf = clf

        self.X = X
        self.y = y

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver
        self.clf.fit(self.X, y=self.y)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function are the arguments of `Objective.compute`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return self.clf

    def warmup_solver(self):
        pass
