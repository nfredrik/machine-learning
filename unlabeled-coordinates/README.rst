=====
About
=====

This is a demo of unsupervised learning where I try to cluster
(x, y) coordinates.
The problem is very similar to
`this one <http://povilasb.com/misc/machine_learning_intro.html>`_.
Except right now I have a bunch of unlabeled coordinates that I want to
cluster::

        o        ^
            o    |
          o      |
                 |
                 |
                 |
                 |
                 |    x
                 |       x
                 |  x
     ------------+------------->

Data
====

There's a script to generate random data::

    $ pyenv/bin/python3 src/generate_data.py --help
    Usage: generate_data.py [OPTIONS]

    Options:
      --samples INTEGER  Number of generated data samples.
      --output TEXT      File to write json data to.
      --help             Show this message and exit.

For example to generate 10000 samples and save the data to out.json use
this command::

    $ pyenv/bin/python3 src/generate_data.py --samples 10000 --output vectors.json


.. rubric:: References

.. [#f1] http://scikit-learn.org/stable/tutorial/statistical_inference/unsupervised_learning.html
.. [#f2] https://www.youtube.com/watch?v=7Qv0cmJ6FsI
.. [#f3] http://bigdata-madesimple.com/possibly-the-simplest-way-to-explain-k-means-algorithm/
