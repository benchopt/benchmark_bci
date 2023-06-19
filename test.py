


#%%
from braindecode.datasets import MOABBDataset
from benchmark_utils import windows_data

dataset_name = "BNCI2014001"
data = MOABBDataset(dataset_name=dataset_name,
                    subject_ids=None)
dataset = windows_data(data, 'MotorImagery')

# %%
