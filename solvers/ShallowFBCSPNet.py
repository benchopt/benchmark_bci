from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import torch
    from braindecode import EEGClassifier
    from braindecode.augmentation import (
        AugmentedDataLoader,
        ChannelsDropout,
        FTSurrogate,
        IdentityTransform,
        SmoothTimeMask,
    )
    from braindecode.models import ShallowFBCSPNet
    from numpy import linspace, pi
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
        "n_epochs": [4],
        "proba": [0.5],

    }

    sampling_strategy = "run_once"

    def set_objective(self, X, y, sfreq):
        """Set the objective information from Objective.get_objective.

        Objective
        ---------
        X: training data for the model
        y: training labels to train the model.
        sfreq: sampling frequency to allow filtering the data.
        """

        # here we want to define a function that gets the data X,y from Moabb
        # and converts it to data accessible for deep learning methods
        lr = self.lr
        weight_decay = self.weight_decay
        batch_size = self.batch_size
        n_epochs = self.n_epochs
        n_classes = len(set(y))
        n_channels = X[0].shape[0]
        n_times = X[0].shape[1]

        # For the fakedataset, trials are too small for the model with default
        # pool_time_lenght=75, use a smaller one.
        pool_time_length = min(75, n_times // 2)
        model = ShallowFBCSPNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=n_times,
            pool_time_length=pool_time_length,
            final_conv_length="auto",
            add_log_softmax=False
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
                for second in linspace(0.1, 2, 10)
            ]

        elif self.augmentation == "ChannelDropout":
            transforms = [
                ChannelsDropout(
                    probability=self.proba, p_drop=prob, random_state=seed
                )
                for prob in linspace(0, 1, 10)
            ]

        elif self.augmentation == "FTSurrogate":
            transforms = [
                FTSurrogate(
                    probability=self.proba,
                    phase_noise_magnitude=prob,
                    random_state=seed,
                )
                for prob in linspace(0, 2 * pi, 10)
            ]
        else:
            transforms = [IdentityTransform()]

        clf = EEGClassifier(
            model,
            iterator_train=AugmentedDataLoader,
            iterator_train__transforms=transforms,
            criterion=torch.nn.CrossEntropyLoss,
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
                    LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1),
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
        """Return the model to `Objective.evaluate_result`.

        Result
        ------
        model: an instance of a fitted model.
            This model should have methods `score` and `predict`, that accept
            braindecode.WindowsDataset as input.
        """
        return dict(model=self.clf)
