#!/usr/bin/env python

import sys
import argparse
import factorgraph
from factorgraph import FactorGraph
from dataloading import *
from numbskulltypes import *
import numpy as np


class NumbSkull(object):
    """
    Main class for numbskull.
    """
    def __init__(self, **kwargs):
        # Default arguments
        arg_defaults = {
            "directory": None,
            "metafile": None,
            "weightfile": None,
            "variablefile": None,
            "factorfile": None,
            "nthreads": 1,
            "n_learning_epoch": 0,
            "n_inference_epoch": 0,
            "burn_in": 0,
            "stepsize": 0.01,
            "decay": 0.95,
            "regularization": "l2",
            "reg_param": 1,
            "sample_evidence": False,
            "learn_non_evidence": False,
            "quiet": True,
            "verbose": False,
            "version": "0.0"
        }

        for (arg, default) in arg_defaults.iteritems():
            setattr(self, arg, kwargs.get(arg, default))

        self.factorGraphs = []

    def loadFactorGraph(self, weight, variable, factor, equalPredicate, edges,
                        var_copies=1, weight_copies=1):
        print("Not fully implemented yet")
        return

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
        print_info = self.quiet
        print_only_meta = self.verbose

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
                print("        weight: ", w["weight"])
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
                                          diagnostics=self.quiet)

    def learning(self, fgID=0):
        burn_in = self.burn_in
        n_learning_epoch = self.n_learning_epoch
        stepsize = self.stepsize
        regularization = self.regularization
        reg_param = self.reg_param

        self.factorGraphs[fgID].learn(burn_in, n_learning_epoch, stepsize, regularization, reg_param, diagnostics=self.quiet)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Runs a Gibbs sampler",
        epilog="")

    parser.add_argument("directory",
                        metavar="DIRECTORY",
                        nargs="?",
                        help="specify directory of factor graph files",
                        default=".",
                        type=str)
    parser.add_argument("-m", "--meta",
                        metavar="META_FILE",
                        dest="meta",
                        default="graph.meta",
                        type=str,
                        help="meta file")
    # TODO: print default for meta, weight, variable, factor in help
    parser.add_argument("-w", "--weight",
                        metavar="WEIGHTS_FILE",
                        dest="weight",
                        default="graph.weights",
                        type=str,
                        help="weight file")
    parser.add_argument("-v", "--variable",
                        metavar="VARIABLES_FILE",
                        dest="variable",
                        default="graph.variables",
                        type=str,
                        help="variable file")
    parser.add_argument("-f", "--factor",
                        metavar="FACTORS_FILE",
                        dest="factor",
                        default="graph.factors",
                        type=str,
                        help="factor file")
    parser.add_argument("-l", "--n_learning_epoch",
                        metavar="NUM_LEARN_EPOCHS",
                        dest="learn",
                        default=0,
                        type=int,
                        help="number of learning epochs")
    parser.add_argument("-i", "--n_inference_epoch",
                        metavar="NUM_INFERENCE_EPOCHS",
                        dest="inference",
                        default=0,
                        type=int,
                        help="number of inference epochs")
    parser.add_argument("-s", "--stepsize",
                        metavar="LEARNING_STEPSIZE",
                        dest="stepsize",
                        default=0.01,
                        type=float,
                        help="stepsize for learning")
    parser.add_argument("-d", "--decay",
                        metavar="LEARNING_DECAY",
                        dest="decay",
                        default=0.95,
                        type=float,
                        help="decay for learning")
    parser.add_argument("-r", "--reg_param",
                        metavar="LEARNING_REGULARIZATION_PARAM",
                        dest="reg_param",
                        default=1.0,
                        type=float,
                        help="regularization parameter for learning")
    parser.add_argument("-p", "--regularization",
                        metavar="REGULARIZATION",
                        dest="regularization",
                        default="l2",
                        type=str,
                        help="regularization (l1 or l2)")
    parser.add_argument("-b", "--burn_in",
                        metavar="BURN_IN",
                        dest="burn_in",
                        default=0,
                        type=int,
                        help="number of burn-in epochs")
    parser.add_argument("-t", "--threads",
                        metavar="NUM_THREADS",
                        dest="threads",
                        default=1,
                        type=int,
                        help="number of threads per copy")
    parser.add_argument("-q", "--quiet",
                        dest="quiet",
                        default=False,
                        action="store_true",
                        help="quiet")
    parser.add_argument("--sample_evidence",
                        dest="sample_evidence",
                        default=False,
                        action="store_true",
                        help="sample evidence")
    parser.add_argument("--learn_non_evidence",
                        dest="learn_non_evidence",
                        default=False,
                        action="store_true",
                        help="learn non evidence")
    # TODO: verbose option (print all info)
    parser.add_argument("--verbose",
                        dest="verbose",
                        default=False,
                        action="store_true",
                        help="verbose")
    parser.add_argument("--version",
                        action='version',
                        version="%(prog)s 0.0",
                        help="print version number")

    # Initialize NumbSkull #
    args = parser.parse_args(argv)
    ns = NumbSkull(**vars(args))
    ns.loadFGFromFile()
    return ns

if __name__ == "__main__":
    main()
