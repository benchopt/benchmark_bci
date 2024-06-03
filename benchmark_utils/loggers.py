from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skorch.callbacks import LRScheduler, EarlyStopping, EpochScoring


def import_wandb_loggers(
    dataset_name=None, model_name=None, validation_name=None, project_name=None
):
    try:
        from wandb import init, login
        from skorch.callbacks import WandbLogger
        import time
        import os

        os.environ["WANDB_SILENT"] = "true"
        # Hidden this key latter
        login(key="d4c4b9c56bda8a814e122301ad70b0d38f014728")
        # Time is necessary to avoid name conflicts in the cross-subject folds
        ts = time.time()
        name = f"{model_name}-{validation_name}-{dataset_name}-{ts}"

        wandb_run = init(
            project=project_name,
            name=name,
            reinit=True,
            group=f"{model_name}",
        )

        wandb_logger = WandbLogger(wandb_run, save_model=False)
        return wandb_logger

    except ImportError:
        return None


def turn_off_warnings():
    import warnings

    warnings.filterwarnings("ignore")
    import mne

    mne.set_log_level("ERROR")


def get_braindecode_callbacks(
    patience,
    max_epochs,
    dataset_name=None,
    model_name=None,
    validation_name=None,
    project_name=None,
    use_early_stopping=True,
):
    """
    Get the Braindecode callbacks.
    Parameters:
    -----------

    patience: int
        The patience for the EarlyStopping callback.
    max_epochs:
        The maximum number of epochs.

    Returns:
    --------
    callbacks: list
        List of Braindecode callbacks.

    """
    train_bal_acc = EpochScoring(
        scoring="balanced_accuracy",
        on_train=True,
        name="train_bal_acc",
        lower_is_better=False,
    )
    train_acc = EpochScoring(
        scoring="accuracy",
        on_train=True,
        name="train_acc",
        lower_is_better=False,
    )
    valid_bal_acc = EpochScoring(
        scoring="balanced_accuracy",
        on_train=False,
        name="valid_bal_acc",
        lower_is_better=False,
    )

    wandb_callback = import_wandb_loggers(
        dataset_name=dataset_name,
        model_name=model_name,
        validation_name=validation_name,
        project_name=project_name,
    )

    callbacks = [
        ("train_acc", train_acc),
        ("train_bal_acc", train_bal_acc),
        ("valid_bal_acc", valid_bal_acc),
        LRScheduler("CosineAnnealingLR", T_max=max_epochs - 1),
    ]
    if wandb_callback is not None:
        callbacks = callbacks + [wandb_callback]
    if use_early_stopping:
        callbacks = callbacks + [
            EarlyStopping(
                monitor="valid_loss", patience=patience, load_best=True
            )
        ]

    return callbacks
