
Brain Computer Interface (BCI) Benchmark
========================================
|Build Status| |Python 3.8+|

This repository is dedicated to evaluate BCI methods on various task, datasets and paradigm.
The problem which is tackled depends on the paradigm and the evaluation.

BCI problems aims at discriminate between various active conditions of a subjects that is recorded
using a neuroimaging device such as a EEG headband. The paradigm defines the various conditions,
or classes $\{c_1, \dots, c_k\}$ to recognize in the data, for instance the `left-right` paradigm
corresponds to thinking about moving the left or right hand.
Each occurence of a condition is called an epoch or a trial, associated with a short period of time
in the recorded signal $X_i$. The goal of BCI is to learn a classifier able to predict from $X_i$
what is the associated condition $y_i \in \{c_1, \dots, c_k\}$.

Usually, multiple subjects are recorded for multiple sessions, during which they repeat several
time the conditions. The choice of evaluation process define which of these subjects and sessions
are the training data and which are the test data:

- With the `intra-session` paradigm, we use part of the trials from one session as the training
  data and the remaining trials as test data.
- With the `inter-session` evaluation, we use all the trials from various sessions for the same
  subject as training and evaluate on the trials from a different session but for the same subject.
  This makes it harder as the EEG device has moved between the two sessions, but it is still the
  same individual who's recorded in the machine.
- The `inter-subject` evaluation is even harder, as this time, one uses all the trials from different
  subjects to train the classifier and evaluate on the trials from a subject that was not included
  in the training data. This tests the generalization capabilites of the algorithms.

For each of this paradigm, we can evaluate all combinations of train/test trials, sessions or subjects.
Finally, once process to obtain test data $\mathcal D_{test}^i$ from the full data $\mathcal D$ has
been defined, the core metric of the benchmark is the balanced accuracy:

$$\\sum_{\\mathcal D_{test}^i}\\frac1{|\\mathcal D_{test}^i|}\sum_{(X_i, y_i) \\in \\mathcal D_{test}^i} p(y_i)1\\{y_i = f_{\\theta}(X_i)\\}$$


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


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_bci/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_bci/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/release/python-380/
