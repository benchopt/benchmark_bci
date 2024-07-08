from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from pathlib import Path

    from optuna.distributions import (
        FloatDistribution,
        CategoricalDistribution,
        IntDistribution,
    )
    from optuna_integration import OptunaSearchCV

    from skorch.helper import SliceDataset, to_numpy
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline, FunctionTransformer

    from benchmark_utils.pipeline import (
        parser_pipelines,
    )

# Getting the path base on the file.
pipeline_folder = str(Path(__file__).parent.parent / "pipelines")


class Solver(BaseSolver):
    name = "MOABBPipelinesOptuna"
    parameters = {
        "pipeline": [
            "Aug-Cov_reg-Tang-SVM",
            # 'Cov-CSP-LDA_shr', Not working, fix this later
            # 'Cov-CSP-LDA_svd', Not working, fix this later
            "Cov-FgMDM",
            "Cov-MDM",
            # 'Cov-MDMAug', Not working, contact Chris later
            "Cov-Tang-LogReg",
            "Cov-Tang-LogReg_ElNet",
            "Cov-Tang-SVM",
            "Cov-TRCSP-LDA",
            "DUMMY",
            "LogVar-LDA",
            "LogVar-SVM",
        ],
        "n_trials": [50],  # Number of trials to run
        "timeout": [10 * 60],  # 10 minutes
        "n_jobs": [1],
        "n_sub_splits": [5],
    }

    sampling_strategy = "run_once"

    def set_objective(self, X, y, sfreq, extra_info):
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
        self.model = make_pipeline(
            FunctionTransformer(to_numpy),
            parser_pipelines(pipeline_folder)[self.pipeline],
        )
        param = self.sample_parameters()
        params = {
            f"pipeline__{step_name}__{p}": v
            for step_name, step in param.items()
            for p, v in step.items()
        }
        self.clf = OptunaSearchCV(
            estimator=clone(self.model),
            n_trials=self.n_trials,
            refit=True,
            cv=self.n_sub_splits,
            n_jobs=self.n_jobs,
            scoring="balanced_accuracy",
            return_train_score=True,
            param_distributions=params,
            timeout=self.timeout,
        )

    def sample_parameters(self):
        return get_hyperparams_from_pipeline(self.pipeline)

    def run(self, _):

        if isinstance(self.y, SliceDataset):
            self.y = to_numpy(self.y)

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


def fetch_layer_params(name) -> dict:
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
        svm_C = FloatDistribution(1e-6, 1e6, log=True)
        svm_kernel = CategoricalDistribution(["linear", "rbf"])
        svc = dict(C=svm_C, kernel=svm_kernel)
        return {"svc": svc}
    if name == "COV":
        # Parameters for the covariance matrix
        cov_estimator = CategoricalDistribution(["cov", "hub", "lwf", "oas"])
        cov = dict(estimator=cov_estimator)
        return {"covariances": cov}
    if name == "COV_reg".upper():
        # Parameters for the covariance with regularization
        cov_estimator = CategoricalDistribution(["lwf", "oas"])
        cov = dict(estimator=cov_estimator)
        return {"covariances": cov}
    if name == "LDA":
        # Parameters for the LDA
        shrinkage = FloatDistribution(0, 1)
        solver = CategoricalDistribution(["lsqr", "eigen"])
        lda = dict(shrinkage=shrinkage, solver=solver)
        return {"lineardiscriminantanalysis": lda}
    if name == "MDM":
        metric_options = ["riemann", "logeuclid", "euclid"]
        metric = CategoricalDistribution(metric_options)
        mdm = dict(metric=metric)
        return {"mdm": mdm}
    if name == "TRCSP":
        n_filters = (6, 10)
        trcsp = dict(nfilter=n_filters)
        return {"trcsp": trcsp}
    if name == "AUG":
        lag = IntDistribution(1, 10)
        order = IntDistribution(1, 10)
        aug = dict(lag=lag, order=order)
        return {"augmenteddataset": aug}
    if name == "CSP":
        # Parameters for the CSP
        nfilter = IntDistribution(6, 10)
        metric = CategoricalDistribution(["euclid"])
        log = CategoricalDistribution([True, False])
        csp = dict(nfilter=nfilter, metric=metric, log=log)
        return {"csp": csp}
    if name == "FGMDM":
        metric_options = ["riemann", "logeuclid", "euclid"]
        metric = CategoricalDistribution(metric_options)
        fgmdm = dict(metric=metric)
        return {"fgmdm": fgmdm}
    if name == "LOGREG":
        # Optuna parameters for the Logistic Regression
        penalty = CategoricalDistribution(["l2"])
        C = FloatDistribution(1e-6, 1e6, log=True)
        solver = CategoricalDistribution(["lbfgs", "saga"])
        logreg = dict(penalty=penalty, C=C, solver=solver)
        return {"logisticregression": logreg}
    if name == "LogReg_ElNet".upper():
        l1_ratio = FloatDistribution(0.2, 0.75)
        logreg = dict(l1_ratio=l1_ratio)
        return {"logisticregression": logreg}
    if name == "Tang".upper():
        metric = CategoricalDistribution(["riemann"])
        tangentspace = dict(metric=metric)
        return {"tangentspace": tangentspace}

    return {}  # returning void


def get_hyperparams_from_pipeline(pipeline):
    """
    Get the parameters from a pipeline.
    """
    steps = pipeline.split("-")

    param = merge_params_from_steps(steps)

    return param


def merge_params_from_steps(steps):
    """
    Merge parameters from all steps in a pipeline.
    """
    param_list = []
    for step in steps:
        param_list.append(fetch_layer_params(step))

    merged_params = {k: v for params in param_list for k, v in params.items()}
    return merged_params
