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


def get_hyperparams_from_pipeline(pipeline, trial):
    """
    Get the parameters from a pipeline.
    """
    if pipeline == "DUMMY":
        strategy_options = ["most_frequent", "prior", "stratified", "uniform"]
        strategy = trial.suggest_categorical("strategy", strategy_options)
        param = dict(dummyclassifer=dict(strategy=strategy))
    elif pipeline == "MDM":
        # This method is COV -> MDM
        # Covariance matrix parameters
        estimator = trial.suggest_categorical("estimator", ["cov", "lwf"])
        # MDM parameters
        metric_options = ["riemann", "logeuclid"]
        metric = trial.suggest_categorical("metric", metric_options)
        param = dict(
            mdm=dict(metric=metric), covariances=dict(estimator=estimator)
        )
    elif pipeline == "TRCSPLDA":
        estimator = trial.suggest_categorical("estimator", ["cov", "lwf"])
        n_filters = trial.suggest_int("n_filters", 6, 10)
        shrinkage = trial.suggest_float("shrinkage", 0, 1)
        solver = trial.suggest_categorical("solver", ["eigen"])

        param = dict(
            lineardiscriminantanalysis=dict(
                shrinkage=shrinkage, solver=solver
            ),
            covariances=dict(estimator=estimator),
            trcsp=dict(nfilter=n_filters),
        )

    elif pipeline == "TangentSpaceSVMGrid":
        # This method is:
        # COV -> TANGENTSPACE -> SVC
        # Parameters for the covariance matrix
        cov_estimator = ["corr", "cov", "hub", "lwf"]

        cov_estimator = trial.suggest_categorical(
            "cov_estimator", cov_estimator
        )

        # Parameters for the SVC
        svm_C = trial.suggest_float("svm_C", 1e-6, 1e6, log=True)
        svm_kernel = trial.suggest_categorical("svm_kernel", ["linear", "rbf"])

        param = dict(
            covariances=dict(estimator=cov_estimator),
            svc=dict(C=svm_C, kernel=svm_kernel),
        )
    elif pipeline == "AUGTangSVMGrid":
        # This method is:
        # AUGMENTATION -> COV -> TANGENTSPACE -> SVC
        # Parameters for the augmentation
        lag = trial.suggest_int("lag", 1, 10)
        order = trial.suggest_int("order", 1, 10)

        # Parameters for the SVC
        svm_C = trial.suggest_float("svm_C", 1e-10, 1e10, log=True)
        svm_kernel = trial.suggest_categorical("svm_kernel", ["linear", "rbf"])

        param = dict(
            augmenteddataset=dict(order=order, lag=lag),
            svc=dict(C=svm_C, kernel=svm_kernel),
        )
    elif pipeline == "COVCSPLDA":
        # This method is:
        # COV -> CSP -> LDA
        # Parameters for the covariance matrix
        cov_estimator = ["cov", "hub", "lwf"]
        cov_estimator = trial.suggest_categorical(
            "cov_estimator", cov_estimator
        )

        # Parameters for the CSP
        nfilter = trial.suggest_int("nfilter", 6, 10)
        metric = trial.suggest_categorical("metric", ["euclid"])
        log = trial.suggest_categorical("log", [True, False])

        param = dict(
            covariances=dict(estimator=cov_estimator),
            csp=dict(nfilter=nfilter, metric=metric, log=log),
            lineardiscriminantanalysis=dict(solver="lsqr", shrinkage="auto"),
        )
    return param
