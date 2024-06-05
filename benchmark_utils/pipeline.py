from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from moabb.pipelines.utils import parse_pipelines_from_directory


def parser_pipelines(dir_path="pipelines", filtering_paradigm=None):
    """
    Read pipelines from a directory and return a list of pipelines.
    """
    if filtering_paradigm is None:
        filtering_paradigm = ["LeftRightImagery", "MotorImagery"]

    pipeline_configs = parse_pipelines_from_directory(dir_path)
    pipelines = {}
    for pipeline in pipeline_configs:

        if any(par in filtering_paradigm for par in pipeline["paradigms"]):
            pipelines.update({pipeline["name"]: pipeline["pipeline"]})
    return pipelines


def fetch_layer_params(name, trial) -> dict:
    """
    Get the hyperparameters for a pipeline-layer.

    Parameters
    ----------
    name: str
        Name of the pipeline-layer.
    trial: optuna.Trial
        Optuna trial object.

    Returns
    -------
    dict
        Dictionary of Dictionary with the hyperparameters for the pipeline-layer.
    """
    name = name.upper()
    if name == "SVM":
        # Parameters for the SVC
        svm_C = trial.suggest_float("svm_C", 1e-6, 1e6, log=True)
        svm_kernel = trial.suggest_categorical("svm_kernel", ["linear", "rbf"])
        svc = dict(C=svm_C, kernel=svm_kernel)
        return {"svc": svc}
    if name == "COV":
        # Parameters for the covariance matrix
        cov_estimator = trial.suggest_categorical(
            "cov_estimator", ["cov", "hub", "lwf", "oas"]
        )
        cov = dict(estimator=cov_estimator)
        return {"covariances": cov}
    if name == "LDA":
        # Parameters for the LDA
        shrinkage = trial.suggest_float("shrinkage", 0, 1)
        solver = trial.suggest_categorical("solver", ["lsqr", "eigen"])
        lda = dict(shrinkage=shrinkage, solver=solver)
        return {"lineardiscriminantanalysis": lda}
    if name == "MDM":
        metric_options = ["riemann", "logeuclid", "euclid"]
        metric = trial.suggest_categorical("metric", metric_options)
        mdm = dict(metric=metric)
        return {"mdm": mdm}
    if name == "TRCSP":
        n_filters = trial.suggest_int("n_filters", 6, 10)
        trcsp = dict(nfilter=n_filters)
        return {"trcsp": trcsp}
    if name == "AUG":
        lag = trial.suggest_int("lag", 1, 10)
        order = trial.suggest_int("order", 1, 10)
        aug = dict(lag=lag, order=order)
        return {"augmenteddataset": aug}
    if name == "CSP":
        # Parameters for the CSP
        nfilter = trial.suggest_int("nfilter", 6, 10)
        metric = trial.suggest_categorical("metric", ["euclid"])
        log = trial.suggest_categorical("log", [True, False])
        csp = dict(nfilter=nfilter, metric=metric, log=log)
        return {"csp": csp}
    if name == "FGMDM":
        metric_options = ["riemann", "logeuclid", "euclid"]
        metric = trial.suggest_categorical("metric", metric_options)
        fgmdm = dict(metric=metric)
        return {"fgmdm": fgmdm}
    if name == "LOGREG":
        # Optuna parameters for the Logistic Regression
        penalty = trial.suggest_categorical("penalty", ["l2"])
        C = trial.suggest_float("C", 1e-6, 1e6, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        logreg = dict(penalty=penalty, C=C, solver=solver)
        return {"logisticregression": logreg}
    if name == "LogReg_ElNet".upper():
        l1_ratio = trial.suggest_float("l1_ratio", 0.2, 0.75)
        logreg = dict(l1_ratio=l1_ratio)
        return {"logisticregression": logreg}
    if name == "Tang".upper():
        metric = trial.suggest_categorical("metric", ["riemann"])
        tangentspace = dict(metric=metric)
        return {"tangentspace": tangentspace}


def get_hyperparams_from_pipeline(pipeline, trial):
    """
    Get the parameters from a pipeline.
    """
    steps = pipeline.split("-")

    param = merge_params_from_steps(steps, trial)

    return param


def merge_params_from_steps(steps, trial):
    """
    Merge parameters from all steps in a pipeline.
    """
    param_list = []
    for step in steps:
        param_list.append(fetch_layer_params(step, trial))

    merged_params = {k: v
                     for params in param_list
                     for k, v in params.items()}
    return merged_params
