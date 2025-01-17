import sys  # noqa: F401

import pytest  # noqa: F401


def check_test_dataset_get_data(benchmark, dataset_class):
    """Hook called in `test_solver_install`.

    If one solver needs to be skip/xfailed on some
    particular architecture, call pytest.xfail when
    detecting the situation.
    """
    if "MOABBDatasets" in dataset_class.name:
        pytest.skip("MOABBDatasets is not planed to run in CI.")


def check_test_solver_install(solver_class):
    if "green" in solver_class.name.lower():
        pytest.skip("Failing install")
