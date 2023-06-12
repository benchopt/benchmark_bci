# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax

def flatten(liste):
    return [item for sub_list in liste for item in sub_list]


def list_train_test(test, list):
    list_test = [list[test]]
    list_train = list[:test] + list[test+1:]
    return list_test, list_train
