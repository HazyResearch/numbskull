#!/usr/bin/env python
import sys
import argparse
import factorgraph
from factorgraph import FactorGraph
from dataloading import *
from numbskulltypes import *
import numpy as np

class NumbSkull(object):

    def __init__(self, args=None):
        self.args = args
        self.factorGraphs = []

    def loadFGFromFile(self, var_copies=1, weight_copies=1):
        # init necessary input arguments
        directory       = self.args.directory
        metafile        = self.args.meta
        weightfile      = self.args.weight
        variablefile    = self.args.variable
        factorfile      = self.args.factor
        print_info      = self.args.quiet
        print_only_meta = self.args.verbose

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

        load_weights(weight_data, meta["weights"], weight) # NUMBA-based function. Defined in dataloading.py
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
        load_variables(variable_data, meta["variables"], variable) #NUMBA-based method. Defined in dataloading.py
        if print_info and not print_only_meta:
            print("Variables:")
            for (i, v) in enumerate(variable):
                print("    variableId:", i)
                print("        isEvidence:  ", v["isEvidence"])
                print("        initialValue:", v["initialValue"])
                print("        dataType:    ", v["dataType"], "(", dataType(v["dataType"]), ")")
                print("        cardinality: ", v["cardinality"])
                print()

        # load factors
        factor_data = np.memmap(directory + "/" + factorfile, mode="c")
        factor = np.empty(meta["factors"], Factor)
        fstart = np.zeros(meta["factors"] + 1, np.int64)
        fmap = np.zeros(meta["edges"], np.int64)
        equalPredicate = np.zeros(meta["edges"], np.int32)
        load_factors(factor_data, meta["factors"], factor, fstart, fmap, equalPredicate) #Numba-based method. Defined in dataloading.py

        # generate variable-to-factor map
        vstart = np.zeros(meta["variables"] + 1, np.int64)
        vmap = np.zeros(meta["edges"], np.int64)
        compute_var_map(fstart, fmap, vstart, vmap) #Numba-based method. Defined in dataloading.py

        fg = FactorGraph(weight, variable, factor, fstart, fmap, vstart, vmap, equalPredicate, var_copies, weight_copies, len(self.factorGraphs), self.args.threads)
        self.factorGraphs.append(fg)

    def getFactorGraph(self, fgID=0):
        return self.factorGraphs[fgID]

    def inference(self,fgID=0):
        burnin = self.args.burnin
        epochs = self.args.inference

        self.factorGraphs[fgID].inference(burnin,epochs,diagnostics=True)

    def learning(self,fgID=0):
        burnin = self.args.burnin
        learning_epochs = self.args.learn
        stepsize = self.args.stepsize

        self.factorGraphs[fgID].learn(burnin,learning_epochs,stepsize,diagnostics=True)


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
                        help="meta file") # TODO: print default for meta, weight, variable, factor in help
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
    parser.add_argument("-l", "--learn",
                        metavar="NUM_LEARN_EPOCHS",
                        dest="learn",
                        default=0,
                        type=int,
                        help="number of learning epochs")
    parser.add_argument("-i", "--inference",
                        metavar="NUM_INFERENCE_EPOCHS",
                        dest="inference",
                        default=0,
                        type=int,
                        help="number of inference epochs")
    parser.add_argument("-s", "--stepsize",
                        metavar="LEARNING_STEPSIZE",
                        dest="stepsize",
                        default=0,
                        type=float,
                        help="stepsize for learning")
    parser.add_argument("-b", "--burnin",
                        metavar="BURN_IN",
                        dest="burnin",
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
                        #metavar="QUIET",
                        dest="quiet",
                        default=False,
                        action="store_true",
                        #type=bool,
                        help="quiet")
    # TODO: verbose option (print all info)
    parser.add_argument("--verbose",
    #                    metavar="VERBOSE",
                        dest="verbose",
                        default=False,
                        action="store_true",
    #                    type=bool,
                        help="verbose")
    parser.add_argument("--version",
                        action='version',
                        version="%(prog)s 0.0",
                        help="print version number")

    ## Initialize NumbSkull ##
    args = parser.parse_args(argv)
    ns  = NumbSkull(args)
    return ns

if __name__ == "__main__":
    main()

