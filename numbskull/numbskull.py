#!/usr/bin/env python

from __future__ import print_function
import sys
import argparse
import factorgraph
from factorgraph import FactorGraph
from dataloading import *
from numbskulltypes import *
import numpy as np


# Define arguments for both parser in main and NumbSkull
arguments = [
    (tuple(['directory']), 
        {'metavar': 'DIRECTORY', 
         'nargs': '?',
         'default': '.',
         'type': str,
         'help': 'specify the directory of factor graph files'}),
    # TODO: print default for meta, weight, variable, factor in help
    (('-m', '--meta'), 
        {'metavar': 'META_FILE',
         'dest': 'metafile',
         'default': 'graph.meta',
         'type': str,
         'help': 'factor graph metadata file'}),
    (('-w', '--weight'),
        {'metavar': 'WEIGHTS_FILE',
         'dest': 'weightfile',
         'default': 'graph.weights',
         'type': str,
         'help': 'factor weight file'}),
    (('-v', '--variable'),
        {'metavar': 'VARIABLES_FILE',
         'dest': 'variablefile',
         'default': 'graph.variables',
         'type': str,
         'help': 'factor graph variables file'}),
    (('-f', '--factor'),
        {'metavar': 'FACTORS_FILE',
         'dest': 'factorfile',
         'default': 'graph.factors',
         'type': str,
         'help': 'factor file'}),
    (('-l', '--n_learning_epoch'),
        {'metavar': 'NUM_LEARNING_EPOCHS',
         'dest': 'n_learning_epoch',
         'default': 0,
         'type': int,
         'help': 'number of learning epochs'}),
    (('-i', '--n_inference_epoch'),
        {'metavar': 'NUM_INFERENCE_EPOCHS',
         'dest':'n_inference_epoch',
         'default': 0,
         'type': int,
         'help': 'number of inference epochs'}),
    (('-s', '--stepsize'),
        {'metavar': 'LEARNING_STEPSIZE',
         'dest': 'stepsize',
         'default': 0.01,
         'type': float,
         'help': 'stepsize for learning'}),
    (('-d', '--decay'),
        {'metavar': 'LEARNING_DECAY',
         'dest': 'decay',
         'default': 0.95,
         'type': float,
         'help': 'decay for updating stepsize during learning'}),
    (('-r', '--reg_param'),
        {'metavar': 'LEARNING_REGULARIZATION_PARAM',
         'dest': 'reg_param',
         'default': 1.0,
         'type': float,
         'help': 'regularization penalty'}),
    (tuple(['--regularization']),
        {'metavar': 'REGULARIZATION',
         'dest': 'regularization',
         'default': 2,
         'type': int,
         'help': 'regularization (l1 or l2)'}),
    (('-b', '--burn_in'),
        {'metavar': 'BURN_IN',
         'dest':'burn_in',
         'default': 0,
         'type': int,
         'help': 'number of burn-in epochs'}),
    (('-t', '--threads'),
        {'metavar': 'NUM_THREADS',
         'dest':'nthreads',
         'default': 1,
         'type': int,
         'help': 'number of threads to be used'})
]

flags = [
    (tuple(['--sample_evidence']),
        {'default': False,
         'action': 'store_true',
         'help': 'sample evidence variables'}),
    (tuple(['--learn_non_evidence']),
        {'default': False,
         'action': 'store_true',
         'help': 'learn from non-evidence variables'}),
    (tuple(['--quiet']),
        {'default': False,
         'action': 'store_true',
         'help': 'quiet'}),
    (tuple(['--verbose']),
        {'default': False,
         'action': 'store_true',
         'help': 'verbose'})
]


class NumbSkull(object):
    """
    Main class for numbskull.
    """

    def __init__(self, **kwargs):
        # Set version
        self.version = "0.0"
        # Initialize default execution arguments
        arg_defaults = {}
        for arg, opts in arguments:
            if 'directory' in arg[0]:
                arg_defaults['directory'] = opts['default']
            else:
                arg_defaults[opts['dest']] = opts['default']
        # Initialize default execution flags
        for arg, opts in flags:
            arg_defaults[arg[0].strip('--')] = opts['default']
        for (arg, default) in arg_defaults.iteritems():
            setattr(self, arg, kwargs.get(arg, default))

        self.factorGraphs = []

    def loadFactorGraph(self, weight, variable, factor, fstart, fmap,
                        equalPredicate, edges, var_copies=1, weight_copies=1):
        # Assert input arguments correspond to NUMPY arrays
        assert(type(weight) == np.ndarray and weight.dtype == Weight)
        assert(type(variable) == np.ndarray and variable.dtype == Variable)
        assert(type(factor) == np.ndarray and factor.dtype == Factor)
        assert(type(equalPredicate) == np.ndarray and
               equalPredicate.dtype == np.int32)
        assert(type(edges) == int or type(edges) == np.int64)

        # Initialize metadata
        meta = {}
        meta['weights'] = weight.shape[0]
        meta['variables'] = variable.shape[0]
        meta['factors'] = factor.shape[0]
        meta['edges'] = edges

        # generate variable-to-factor map
        vstart = np.zeros(meta["variables"] + 1, np.int64)
        vmap = np.zeros(meta["edges"], np.int64)

        # Numba-based method. Defined in dataloading.py
        compute_var_map(fstart, fmap, vstart, vmap)

        fg = FactorGraph(weight, variable, factor, fstart, fmap, vstart, vmap,
                         equalPredicate, var_copies, weight_copies,
                         len(self.factorGraphs), self.nthreads)
        self.factorGraphs.append(fg)

    def loadFGFromFile(self, directory=None, metafile=None, weightfile=None,
                       variablefile=None, factorfile=None, var_copies=1,
                       weight_copies=1):
        # init necessary input arguments
        if not self.directory:
            print("No factor graph specified")
            return
        else:
            directory = self.directory

        metafile = 'graph.meta' if not metafile else metafile
        weightfile = 'graph.weights' if not weightfile else weightfile
        variablefile = 'graph.variables' if not variablefile else variablefile
        factorfile = 'graph.factors' if not factorfile else factorfile
        print_info = not self.quiet
        print_only_meta = not self.verbose

        # load metadata
        meta = np.loadtxt(directory + "/" + metafile,
                          delimiter=',',
                          dtype=Meta)
        meta = meta[()]

        if print_info:
            print("Meta:")
            print("    weights:  ", meta["weights"])
            print("    variables:", meta["variables"])
            print("    factors:  ", meta["factors"])
            print("    edges:    ", meta["edges"])
            print()

        # load weights
        weight_data = np.memmap(directory + "/" + weightfile, mode="c")
        weight = np.empty(meta["weights"], Weight)

        # NUMBA-based function. Defined in dataloading.py
        load_weights(weight_data, meta["weights"], weight)
        if print_info and not print_only_meta:
            print("Weights:")
            for (i, w) in enumerate(weight):
                print("    weightId:", i)
                print("        isFixed:", w["isFixed"])
                print("        weight: ", w["initialValue"])
            print()

        # load variables
        variable_data = np.memmap(directory + "/" + variablefile, mode="c")
        variable = np.empty(meta["variables"], Variable)

        # NUMBA-based method. Defined in dataloading.py
        load_variables(variable_data, meta["variables"], variable)
        if print_info and not print_only_meta:
            print("Variables:")
            for (i, v) in enumerate(variable):
                print("    variableId:", i)
                print("        isEvidence:  ", v["isEvidence"])
                print("        initialValue:", v["initialValue"])
                print("        dataType:    ", v["dataType"],
                      "(", dataType(v["dataType"]), ")")
                print("        cardinality: ", v["cardinality"])
                print()

        # load factors
        factor_data = np.memmap(directory + "/" + factorfile, mode="c")
        factor = np.empty(meta["factors"], Factor)
        fstart = np.zeros(meta["factors"] + 1, np.int64)
        fmap = np.zeros(meta["edges"], np.int64)
        equalPredicate = np.zeros(meta["edges"], np.int32)

        # Numba-based method. Defined in dataloading.py
        load_factors(factor_data, meta["factors"], factor, fstart, fmap,
                     equalPredicate)

        # generate variable-to-factor map
        vstart = np.zeros(meta["variables"] + 1, np.int64)
        vmap = np.zeros(meta["edges"], np.int64)

        # Numba-based method. Defined in dataloading.py
        compute_var_map(fstart, fmap, vstart, vmap)

        fg = FactorGraph(weight, variable, factor, fstart, fmap, vstart, vmap,
                         equalPredicate, var_copies, weight_copies,
                         len(self.factorGraphs), self.nthreads)
        self.factorGraphs.append(fg)

    def getFactorGraph(self, fgID=0):
        return self.factorGraphs[fgID]

    def inference(self, fgID=0):
        burn_in = self.burn_in
        n_inference_epoch = self.n_inference_epoch

        self.factorGraphs[fgID].inference(burn_in, n_inference_epoch,
                                          diagnostics=not self.quiet)

    def learning(self, fgID=0):
        burn_in = self.burn_in
        n_learning_epoch = self.n_learning_epoch
        stepsize = self.stepsize
        regularization = self.regularization
        reg_param = self.reg_param

        self.factorGraphs[fgID].learn(burn_in, n_learning_epoch,
                                    stepsize, regularization, reg_param,
                                    diagnostics=not self.quiet,
                                    learn_non_evidence=self.learn_non_evidence)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Runs a Gibbs sampler",
        epilog="")
    # Add version to parser
    parser.add_argument("--version",
                        action='version',
                        version="%(prog)s 0.0",
                        help="print version number")
    # Add execution arguments to parser
    for arg, opts in arguments:
        parser.add_argument(*arg, **opts)
    # Add flags to parser
    for arg, opts in flags:
        parser.add_argument(*arg, **opts)
    # Initialize NumbSkull #
    args = parser.parse_args(argv)
    ns = NumbSkull(**vars(args))
    ns.loadFGFromFile()
    return ns

if __name__ == "__main__":
    main()
