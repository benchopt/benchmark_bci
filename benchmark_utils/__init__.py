# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax

from .transformation import smooth_timemask, channels_dropout
from .dataset import windows_data, split_windows_train_test
from .augmented_dataset import AugmentedBCISolver
from .augmented_method import Covariances_augm

__all__ = [
    "smooth_timemask",
    "channels_dropout",
    "windows_data",
    "split_windows_train_test",
    "AugmentedBCISolver",
    "Covariances_augm",
]
