from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from pathlib import Path
    from skorch.helper import SliceDataset, to_numpy
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import FunctionTransformer

    from benchmark_utils.pipeline import parser_pipelines

# Getting the path base on the file.
pipeline_folder = str(Path(__file__).parent.parent / 'pipelines')


class Solver(BaseSolver):
    name = "MOABBPipelines"
    parameters = {
        "pipeline": [
            'Aug-Cov-Tang-SVM',
            'Cov-CSP-LDA_shr',
            'Cov-CSP-LDA_svd',
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
        ]
    }

    sampling_strategy = 'run_once'

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
        self.clf = make_pipeline(
            FunctionTransformer(to_numpy),
            parser_pipelines(pipeline_folder)[self.pipeline]
        )

    def run(self, _):
        """Run the solver to evaluate it for a given number of augmentation.

        With this dataset, we consider that the performance curve is sampled
        for various number of augmentation applied to the dataset.
        """
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
