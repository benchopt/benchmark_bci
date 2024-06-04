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


def get_parameters_for_layer(name, trial):
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
        Dictionary with the hyperparameters for the pipeline-layer.
    """
    name = name.upper()
    if name == "SVM":
        # Parameters for the SVC
        svm_C = trial.suggest_float("svm_C", 1e-6, 1e6, log=True)
        svm_kernel = trial.suggest_categorical("svm_kernel", ["linear", "rbf"])
        svc = dict(C=svm_C, kernel=svm_kernel)
        return svc
    if name == "COV":
        # Parameters for the covariance matrix
        cov_estimator = trial.suggest_categorical(
            "cov_estimator", ["cov", "hub", "lwf", "oas"]
        )
        cov = dict(estimator=cov_estimator)
        return cov
    if name == "LDA":
        # Parameters for the LDA
        shrinkage = trial.suggest_float("shrinkage", 0, 1)
        solver = trial.suggest_categorical("solver", ["lsqr", "eigen"])
        lda = dict(shrinkage=shrinkage, solver=solver)
        return lda
    if name == "MDM":
        metric_options = ["riemann", "logeuclid", "euclid"]
        metric = trial.suggest_categorical("metric", metric_options)
        mdm = dict(metric=metric)
        return mdm
    if name == "TRCSP":
        n_filters = trial.suggest_int("n_filters", 6, 10)
        trcsp = dict(nfilter=n_filters)
        return trcsp
    if name == "AUG":
        lag = trial.suggest_int("lag", 1, 10)
        order = trial.suggest_int("order", 1, 10)
        aug = dict(lag=lag, order=order)
        return aug
    if name == "CSP":
        # Parameters for the CSP
        nfilter = trial.suggest_int("nfilter", 6, 10)
        metric = trial.suggest_categorical("metric", ["euclid"])
        log = trial.suggest_categorical("log", [True, False])
        csp = dict(nfilter=nfilter, metric=metric, log=log)
        return csp
    if name == "FGMDM":
        # I am not sure about this!!!

        metric_options = ["riemann", "logeuclid", "euclid"]
        metric = trial.suggest_categorical("metric", metric_options)
        fgmdm = dict(metric=metric)
        return fgmdm
    if name == "LOGREG":
        # Optuna parameters for the Logistic Regression
        penalty = trial.suggest_categorical("penalty", ["l2"])
        C = trial.suggest_float("C", 1e-6, 1e6, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        logreg = dict(penalty=penalty, C=C, solver=solver)
        return logreg
    if name == "LogReg_ElNet".upper():
        l1_ratio = trial.suggest_float("l1_ratio", 0.2, 0.75)
        logreg = dict(l1_ratio=l1_ratio)
        return logreg


def get_hyperparams_from_pipeline(pipeline, trial):
    """
    Get the parameters from a pipeline.
    """
    _ = [
        "Aug-Cov-Tang-SVM",  # ok
        "Cov-CSP-LDA_shr",  # talk with Thomas
        "Cov-CSP-LDA_svd",  # talk with Thomas
        "Cov-FgMDM",  # need to check
        "Cov-MDM",  # okay
        "Cov-Tang-LogReg",  # okay
        "Cov-Tang-LogReg_ElNet",  # okay
        "Cov-Tang-SVM",
        "Cov-TRCSP-LDA",
        "DUMMY",
        "LogVar-LDA",
        "LogVar-SVM",
    ]

    if pipeline == "Aug-Cov-Tang-SVM":
        # Parameters for the augmentation
        aug = get_parameters_for_layer("AUG", trial)
        # Parameters for the SVC
        svm = get_parameters_for_layer("SVM", trial)

        param = dict(
            augmenteddataset=aug,
            svc=svm,
        )
    # Talk with Thomas about these pipelineS
    elif pipeline == "Cov-CSP-LDA_shr" or pipeline == "Cov-CSP-LDA_svd":
        # This method is:
        # COV -> CSP -> LDA
        # Parameters for the covariance matrix
        covariances = get_parameters_for_layer("COV", trial)
        # Parameters for the CSP
        csp = get_parameters_for_layer("CSP", trial)
        # Parameters for the LDA
        lda = get_parameters_for_layer("LDA", trial)

        param = dict(
            covariances=covariances,
            csp=csp,
            lineardiscriminantanalysis=lda,
        )
    elif pipeline == "Cov-FgMDM":
        # This method is:
        # COV -> FgMDM
        # Parameters for the covariance matrix
        covariances = get_parameters_for_layer("COV", trial)
        # Parameters for the Fg MDM
        fgmdm = get_parameters_for_layer("FgMDM", trial)
        # Check if this correct
        param = dict(
            covariances=covariances,
            fgmdm=fgmdm,
        )

    elif pipeline == "Cov-MDM":
        # Covariance matrix parameters
        covariance = get_parameters_for_layer("COV", trial)
        # MDM parameters
        mdm = get_parameters_for_layer("MDM", trial)

        param = dict(mdm=mdm, covariances=covariance)
    elif pipeline == "Cov-Tang-LogReg":
        # This method is:
        # COV -> TANGENTSPACE -> LOGREG
        # Parameters for the covariance matrix
        covariances = get_parameters_for_layer("COV", trial)
        # Parameters for the Logistic Regression
        logreg = get_parameters_for_layer("LogReg", trial)

        param = dict(
            covariances=covariances,
            tangentspace=dict(metric="riemann"),
            logisticregression=logreg,
        )
    elif pipeline == "Cov-Tang-LogReg_ElNet":
        # This method is:
        # COV -> TANGENTSPACE -> LOGREG
        # Parameters for the covariance matrix
        covariances = get_parameters_for_layer("COV", trial)
        # Parameters for the Logistic Regression
        logreg = get_parameters_for_layer("LogReg_ElNet", trial)

        param = dict(
            covariances=covariances,
            logisticregression=logreg,
        )
    elif pipeline == "Cov-TRCSP-LDA":
        covariances = get_parameters_for_layer("COV", trial)
        lda = get_parameters_for_layer("LDA", trial)
        trcsp = get_parameters_for_layer("TRCSP", trial)

        param = dict(
            lineardiscriminantanalysis=lda,
            covariances=covariances,
            trcsp=trcsp,
        )

    elif pipeline == "Cov-Tang-SVM":
        # This method is:
        # COV -> TANGENTSPACE -> SVC
        # Parameters for the covariance matrix
        covariances = get_parameters_for_layer("COV", trial)
        # Parameters for the SVC
        svm = get_parameters_for_layer("SVM", trial)

        param = dict(
            covariances=covariances,
            svc=svm,
        )
    elif pipeline == "LogVar-LDA":
        # This method is:
        # LOGVAR -> LDA
        # Parameters for the LDA
        lda = get_parameters_for_layer("LDA", trial)

        param = dict(
            logvar=dict(),
            lineardiscriminantanalysis=lda,
        )
    elif pipeline == "LogVar-SVM":
        # This method is:
        # LOGVAR -> SVC
        # Parameters for the SVC
        svm = get_parameters_for_layer("SVM", trial)

        param = dict(
            logvar=dict(),
            svc=svm,
        )
    else:
        param = dict()
    return param
