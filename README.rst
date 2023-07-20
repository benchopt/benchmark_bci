
Brain Computer Interface (BCI) Benchmark
========================================
|Build Status| |Python 3.8+|

This repository evaluates BCI methods on various tasks, datasets, and paradigms.
The problem that is tackled depends on the paradigm and the evaluation.

BCI problems aim to discriminate between various active conditions of subjects that are recorded
using a neuroimaging device such as an EEG headband. The paradigm defines the various conditions,
or classes $\\{c_1, \\dots, c_k\\}$ to recognize in the data. For instance, the `left-right` paradigm
corresponds to imagining the moving of the left or right hand.
Each condition occurrence is called an epoch or a trial, associated with a short period of time
in the recorded signal $X_i$. BCI aims to learn a classifier able to predict from $X_i$
with the associated condition $y_i \\in \\{c_1, \\dots, c_k\\}$.

Usually, multiple subjects are recorded for multiple sessions, during which they repeat several
times the conditions. The choice of evaluation process defines which of these subjects and sessions
are the training data and which are the test data:

- With the `intra-session` evaluation (Cross-Session), we use part of the trials from one session as the training
  data and the remaining trials as test data.
- With the `inter-session` evaluation (Within-Session), we use all the trials from various sessions for the same
  subject as training and evaluate on the trials from a different session but for the same subject.
  This makes it harder as the EEG device has moved between the two sessions, but it is still the
  the same individual who's recorded in the machine.
- The `inter-subject` evaluation (Cross-Subject) is even harder, as this time, one uses all the trials from different
  subjects to train the classifier and evaluate the trials from a subject that was not included
  in the training data. This tests the generalization capabilities of the algorithms.

For each of theses paradigms, we can evaluate all combinations of train/test trials, sessions, or subjects.
Finally, once the process to obtain test data $\\mathcal D_{test}^i$ from the full data $\\mathcal D$ has
been defined, the core metric of the benchmark is the balanced accuracy:

$$ \\sum_{\\mathcal D_{test}^i} \\frac{1}{| \\mathcal D_{test}^i|}  \\sum_{(X_i, y_i) \\in \\mathcal D_{test}^i}  p(y_i) 1\\{y_i = f_{\\theta}(X_i)\\}$$ 


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_bci
   $ benchopt run benchmark_bci

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

    $ benchopt run benchmark_bci -s MDM -d BNCI -r 1 -n 1

We recomend to use the pre-commit configuration in the repository to ensure that the code is properly formatted before each commit:

.. code-block::

    $ pip install pre-commit
    $ pre-commit install


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_bci/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_bci/actions
.. |Python 3.8+| image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/release/python-380/
