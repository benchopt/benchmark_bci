from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import warnings
    import mne


def turn_off_warnings():

    warnings.filterwarnings("ignore")
    mne.set_log_level("ERROR")
