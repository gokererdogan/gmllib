"""
gmllib

This file contains classes for running experiments.
We quite often need to run a method with various
different parameters and tweak the method until we
get it working. These classes are intended to make
that process easier.

Goker Erdogan
2015
https://github.com/gokererdogan
"""

import warnings
import time
import itertools as iter
import cPickle as pkl
import pandas as pd

class Experiment(object):
    """
    Experiment class implements functionality for running a given method
    with different sets of parameters and combining the results into a
    single data frame.
    """
    def __init__(self, name, experiment_method, dataset, **params):
        """
        Initialize Experiment instance
        :param name: Experiment name
        :param experiment_method: Method to run. This method should
            return a dictionary.
        :param dataset: Dataset passed to the experiment_method
        :param params: A collection of keyword=value arguments that
        specify the set of parameters to test. The cartesian products
        of all parameters are taken and experiment_method is run for
        each parameter set.
            e.g., lr=[0.1, 0.2], alpha=[1, 3] results in parameter
            set: [{lr: 0.1, alpha: 1}, {lr: 0.1, alpha: 3},
                  {lr: 0.2, alpha: 1}, {lr: 0.2, alpha: 3}]
        :return: Experiment instance
        """
        self.name = name
        self.experiment_method = experiment_method
        self.dataset = dataset
        self.params = Experiment.keyword_args_to_parameter_set(**params)
        self.experiment_finished = False
        self.start_time = None
        self.start_time_str = None
        self.end_time = None
        self.end_time_str = None
        self.results = None

    @staticmethod
    def keyword_args_to_parameter_set(**params):
        """
        Convert a dictionary of keyword arguments to a set of parameters.
        Takes the cartesian product of keyword argument to form the
        parameters.
        :param params: A dictionary of parameters
        :return: A list of dictionaries where each dictionary is one set of
        parameters.
        """
        # check if any of the values are not lists, if so make them lists
        for k, v in params.iteritems():
            if not isinstance(v, list):
                params[k] = [v]
        params_vals = iter.product(*params.values())
        params_set = [dict(zip(params.keys(), l)) for l in params_vals]
        return params_set

    def reset(self):
        self.experiment_finished = False

    def run(self, parallel=False):
        if self.experiment_finished:
            warnings.warn("Experiment is already run. Call reset if you want to re-run the experiment.")
            return

        self.start_time = time.time()
        self.start_time_str = time.strftime("%Y%m%d_%H%M%S")

        # run method with each parameter
        results = []
        for param in self.params:
            start = time.time()
            result = self.experiment_method(dataset=self.dataset, **param)
            end = time.time()
            result.update(param)
            result.update({'StartTime': start, 'EndTime': end})
            results.append(result)

        self.end_time = time.time()
        self.end_time_str = time.strftime("%Y%m%d_%H%M%S")
        self.results = pd.DataFrame(results)
        self.experiment_finished = True

    def save(self, path="."):
        if self.experiment_finished:
            if path.strip() == "":
                path = "."
            pkl.dump(self, open("{0:s}/{1:s}_{2:s}.pkl".format(path, self.name, self.start_time_str), 'wb'))
        else:
            warnings.warn("Experiment is not run yet. Results cannot be saved.")

    def save_csv(self, path="."):
        if self.experiment_finished:
            if path.strip() == "":
                path = "."
            open("{0:s}/{1:s}_{2:s}.csv".format(path, self.name, self.start_time_str), 'w').write(self.results.to_string())
        else:
            warnings.warn("Experiment is not run yet. Results are not available.")
