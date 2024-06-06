from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils import OptunaSolver
    from skorch.helper import to_numpy
    from sklearn.pipeline import make_pipeline, FunctionTransformer

    from benchmark_utils.pipeline import (
        parser_pipelines,
    )


class Solver(OptunaSolver):
    name = "MOABBPipelinesOptuna"
    parameters = {
        "pipeline": [
            'Aug-Cov_reg-Tang-SVM',
            # 'Cov-CSP-LDA_shr', Not working, fix this later
            # 'Cov-CSP-LDA_svd', Not working, fix this later
            'Cov-FgMDM',
            'Cov-MDM',
            # 'Cov-MDMAug', Not working, contact Chris later
            'Cov-Tang-LogReg',
            'Cov-Tang-LogReg_ElNet',
            'Cov-Tang-SVM',
            'Cov-TRCSP-LDA',
            'DUMMY',
            'LogVar-LDA',
            'LogVar-SVM',
        ],
    }

    def get_model(self):
        return make_pipeline(
            FunctionTransformer(to_numpy),
            parser_pipelines()[self.pipeline]
        )

    def sample_parameters(self, trial):
        return get_hyperparams_from_pipeline(self.pipeline, trial)


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
        Dictionary of Dictionary with the hyperparameters for the
        pipeline-layer.
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
    if name == "COV_reg".upper():
        # Parameters for the covariance with regularization
        cov_estimator = trial.suggest_categorical(
            "cov_estimator", ["lwf", "oas"]
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
    return {} # returning void 

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

    merged_params = {k: v for params in param_list for k, v in params.items()}
    return merged_params
