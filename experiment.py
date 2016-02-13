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
import multiprocessing as mp
import os
import itertools as iter
import cPickle as pkl
import pandas as pd

class Experiment(object):
    """
    Experiment class implements functionality for running a given method
    with different sets of parameters and combining the results into a
    single data frame. This class also implements a crude form of parallelism
    that allows the experiment to be run on multiple cores.
    """
    def __init__(self, name, experiment_method, grouped_params=None, **params):
        """
        Initialize Experiment instance
        :param name: Experiment name
        :param experiment_method: Method to run. This method should
            return a dictionary.
        :param grouped_params: The names of the parameters in the same group.
            Theses parameters are treated as a single parameter, i.e., only
            one of the parameters is included in the cartesian product.
                e.g., a=[1, 2], b=[3, 4], c=[-3, -4] with 'b' and 'c'
                grouped together results in parameter set:
                    {'a': 1, 'c': -3, 'b': 3}
                    {'a': 1, 'c': -4, 'b': 4}
                    {'a': 2, 'c': -3, 'b': 3}
                    {'a': 2, 'c': -4, 'b': 4}
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
        self.params = Experiment.keyword_args_to_parameter_set(grouped_params, **params)
        self.experiment_finished = False
        self.start_time = None
        self.start_time_str = None
        self.end_time = None
        self.end_time_str = None
        self.results = None

    @staticmethod
    def keyword_args_to_parameter_set(grouped_params=None, **params):
        """
        Convert a dictionary of keyword arguments to a set of parameters.
        Takes the cartesian product of keyword argument to form the
        parameters.
        :param grouped_params: A list of the names of the parameters grouped together.
        :param params: A dictionary of parameters
        :return: A list of dictionaries where each dictionary is one set of
        parameters.
        """
        # check if any of the values are not lists, if so make them lists
        for k, v in params.iteritems():
            if not isinstance(v, list):
                params[k] = [v]

        if grouped_params is not None:
            # parameters in the group should have the same number of possible values
            gp_len = len(params[grouped_params[0]])
            for gp in grouped_params:
                if len(params[gp]) != gp_len:
                    raise ValueError("All parameters in a group should have the same number of possible values.")

            # separate the dictionary for grouped params
            g_params = {}
            for gp in grouped_params:
                g_params[gp] = params[gp]
                del params[gp]

            # construct the parameter set for grouped parameters
            g_params_vals = zip(*g_params.values())
            g_params_set = [dict(zip(g_params.keys(), l)) for l in g_params_vals]

        # construct the parameter set for non-grouped variables
        ng_params_vals = iter.product(*params.values())
        ng_params_set = [dict(zip(params.keys(), l)) for l in ng_params_vals]


        if grouped_params is not None:
            params_set = []
            for ps in ng_params_set:
                for gps in g_params_set:
                    params_set.append(dict(zip(ps.keys() + gps.keys(), ps.values() + gps.values())))
        else:
            params_set = ng_params_set

        return params_set

    def reset(self):
        self.experiment_finished = False

    def run(self, parallel=False, num_processes=2):
        if self.experiment_finished:
            warnings.warn("Experiment is already run. Call reset if you want to re-run the experiment.")
            return

        self.start_time = time.time()
        self.start_time_str = time.strftime("%Y%m%d_%H%M%S")

        results = []
        if parallel and num_processes > 0:
            # run in parallel
            proc_pool = mp.Pool(processes=num_processes)
            async_results = [proc_pool.apply_async(self.experiment_method, kwds=d) for d in self.params]
            proc_pool.close()
            proc_pool.join()
            for param, ar in zip(self.params, async_results):
                try:
                    result = ar.get()
                except Exception as e:
                    warnings.warn("Experiment warning: One process failed.")
                    print(e)
                else:
                    result.update(param)
                    results.append(ar.get())
        else:
            for param in self.params:
                start = time.time()
                result = self.experiment_method(**param)
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
            self.results.to_csv("{0:s}/{1:s}_{2:s}.csv".format(path, self.name, self.start_time_str))
        else:
            warnings.warn("Experiment is not run yet. Results are not available.")

    def append_csv(self, file):
        """
        Append the results of the experiment to csv file.
        If the file does not exist, it is created.
        """
        if self.experiment_finished:
            if os.path.isfile(file):
                # read the already available results
                df = pd.read_csv(file, index_col=0)
            else:
                df = pd.DataFrame()
            # combine the results
            df = pd.concat([df, self.results], ignore_index=True)
            # write to file
            df.to_csv(file)
        else:
            warnings.warn("Experiment is not run yet. Results are not available.")
