
My Benchopt Benchmark
=====================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of **describe your problem**:


$$\\min_{w} f(X, w)$$


where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features and


$$X \\in \\mathbb{R}^{n \\times p} \\ , \\quad w \\in \\mathbb{R}^p$$


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/tomMoral/benchmark_bci
   $ benchopt run benchmark_bci

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_bci -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10
    $ benchopt run benchmark_bci -s MDM -d BNCI -r 1 -n 1


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/tomMoral/benchmark_bci/workflows/Tests/badge.svg
   :target: https://github.com/tomMoral/benchmark_bci/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
