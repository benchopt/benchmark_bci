from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import torch

    from braindecode import EEGClassifier
    from skorch.dataset import ValidSplit

    from benchmark_utils import turn_off_warnings, get_braindecode_callbacks
    turn_off_warnings()


class Solver(BaseSolver):
    name = "BraindecodeModels"
    parameters = {
        "model": [
            "ATCNet",
            "EEGConformer",
            "EEGInceptionERP",
            "EEGInceptionMI",
            "EEGITNet",
            "EEGNetv4",
            "ShallowFBCSPNet",
            "TIDNet",
            "BIOT",
            "AttentionBaseNet",
            "Labram",
            "EEGSimpleConv",
            "ContraWR",
        ],
        "batch_size": [64],
        "valid_set": [0.2],
        "patience": [50],
        "max_epochs": [100],
        "learning_rate": [10**-4],
        "weight_decay": [0],
        "random_state": [42],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X, y, sfreq, metadata_info):
        """Set the objective information from Objective.get_objective.

        Objective
        ---------
        X: training data for the model
        y: training labels to train the model.
        sfreq: sampling frequency to allow filtering the data.
        """
        self.sfreq = sfreq
        self.X = X
        self.y = y

        device = "cuda" if torch.cuda.is_available() else "cpu"
        n_classes = len(set(y))
        n_chans = X[0].shape[0]
        n_times = X[0].shape[1]

        callbacks = get_braindecode_callbacks(
            patience=self.patience,
            max_epochs=self.max_epochs,
            dataset_name="BCNI",
            model_name=self.model,
            validation_name="InterSubject-cv-5",
            project_name="benchmark_d",
        )
        self.clf = EEGClassifier(
            module=self.model,
            module__n_outputs=n_classes,
            module__n_chans=n_chans,
            module__n_times=n_times,
            module__sfreq=self.sfreq,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=self.learning_rate,
            train_split=ValidSplit(
                cv=self.valid_set,
                stratified=True,
                random_state=self.random_state,
            ),
            # using valid_set for validation
            batch_size=self.batch_size,
            device=device,
            verbose=True,
            max_epochs=self.max_epochs,
            optimizer__weight_decay=self.weight_decay,
            classes=list(range(n_classes)),
            callbacks=callbacks,
        )

    def run(self, _):
        """Run the solver to evaluate.
        """
        self.clf.fit(self.X, self.y)

    def get_result(self):
        """Return the model to `Objective.evaluate_result`.

        Result
        ------
        model: an instance of a fitted model.
            This model should have methods `score` and `predict`, that accept
            braindecode.WindowsDataset as input.
        """
        return dict(model=self.clf)
