from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    from benchmark_utils import OptunaSolver
    from skorch.helper import to_numpy
    from sklearn.pipeline import make_pipeline, FunctionTransformer

    from benchmark_utils.pipeline import (
        parser_pipelines,
        get_hyperparams_from_pipeline,
    )


class Solver(OptunaSolver):
    name = "MOABBPipelinesOptuna"
    parameters = {
        "pipeline": [
            'Aug-Cov-Tang-SVM',
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
