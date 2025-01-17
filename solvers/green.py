from benchopt import BaseSolver
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from benchmark_utils import get_braindecode_callbacks
    from braindecode import EEGClassifier
    from green.research_code.pl_utils import get_green
    from green.wavelet_layers import RealCovariance
    from skorch.dataset import ValidSplit


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.


class Solver(BaseSolver):
    name = "green"
    parameters = {
        "lr": [1e-2],
        "max_epochs": [300],
        "patience": [100],
        "weight_decay": [0],
        "valid_set": [0.2],
        "batch_size": [64],
        "proba": [0.5],
        "bi_out": [[16]],
        "hidden_dim": [[8]],
        "n_freqs": [10],
        "random_state": [0],
        "kernel_width_s": [0.5],
        "dropout": [0.5],

    }

    sampling_strategy = "run_once"
    requirements = [
        "pip:git+https://github.com/Roche/neuro-green",
    ]

    def set_objective(self, X, y, sfreq, extra_info):
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
        n_classes = len(set(y))

        n_channels = X[0].shape[0]

        model = get_green(
            n_freqs=self.n_freqs,
            kernel_width_s=self.kernel_width_s,
            n_ch=n_channels,
            sfreq=sfreq,
            orth_weights=True,
            dropout=self.dropout,
            hidden_dim=self.hidden_dim,
            logref="logeuclid",
            pool_layer=RealCovariance(),
            bi_out=self.bi_out,
            dtype=torch.float32,
            out_dim=n_classes,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        callbacks = get_braindecode_callbacks(
            dataset_name=extra_info["dataset_name"],
            patience=self.patience,
            max_epochs=self.max_epochs,
            model_name="Green",
            validation_name=extra_info["evaluation_process"],
            project_name=f"benchmark_{extra_info['paradigm_name']}",
        )

        clf = EEGClassifier(
            module=model,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=lr,
            train_split=ValidSplit(
                cv=self.valid_set,
                stratified=True,
                random_state=self.random_state,
            ),
            batch_size=self.batch_size,
            device=device,
            verbose=True,
            max_epochs=self.max_epochs,
            optimizer__weight_decay=self.weight_decay,
            classes=list(range(n_classes)),
            callbacks=callbacks,
        )

        self.clf = clf

        self.X = X
        self.y = y

    def run(self, _):
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
