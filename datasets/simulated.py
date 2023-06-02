from benchopt import BaseDataset, safe_import_context
import pickle 


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import moabb
    from moabb.datasets import BNCI2014001, Zhou2016
    from moabb.evaluations import CrossSessionEvaluation
    from moabb.paradigms import LeftRightImagery


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "dataldacsp"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    

   
    
    def get_data(self):
        
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        subj = [1, 2]
        datasets = [Zhou2016(), BNCI2014001()]
        dataset = BNCI2014001()
        for d in datasets:
            d.subject_list = subj
        
        paradigm = LeftRightImagery(fmin=8, fmax=35)

        
        for d in datasets:
            sessions = d.get_data(subjects=[2])
        X, labels, meta = paradigm.get_data(dataset=d, subjects=[2])
     

        # The dictionary defines the keyword arguments for `Objective.set_data`G
        return dict(X=np.array(X),y=np.array(labels))
    
    