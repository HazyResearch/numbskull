#!/usr/bin/env python

"""TODO: This is a docstring."""

from __future__ import print_function, absolute_import
from past.builtins import long
import os
import sys
import argparse
import numbskull.factorgraph
from numbskull.factorgraph import FactorGraph
from numbskull.dataloading import *
from numbskull.numbskulltypes import *
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
    (('-o', '--output_dir'),
        {'metavar': 'OUTPUT_DIR',
         'dest': 'output_dir',
         'default': '.',
         'type': str,
         'help': 'Output dir to contain inference_result.out.text ' +
                 'and inference_result.out.weights.text'}),
    (('-m', '--meta', '--fg_meta'),
        {'metavar': 'META_FILE',
         'dest': 'metafile',
         'default': 'graph.meta',
         'type': str,
         'help': 'factor graph metadata file'}),
    (('-w', '--weight', '--weights'),
        {'metavar': 'WEIGHTS_FILE',
         'dest': 'weightfile',
         'default': 'graph.weights',
         'type': str,
         'help': 'factor weight file'}),
    (('-v', '--variable', '--variables'),
        {'metavar': 'VARIABLES_FILE',
         'dest': 'variablefile',
         'default': 'graph.variables',
         'type': str,
         'help': 'factor graph variables file'}),
    (('-f', '--factor', '--factors'),
        {'metavar': 'FACTORS_FILE',
         'dest': 'factorfile',
         'default': 'graph.factors',
         'type': str,
         'help': 'factor file'}),
    (('--domain', '--domains'),
        {'metavar': 'DOMAINS_FILE',
         'dest': 'domainfile',
         'default': 'graph.domains',
         'type': str,
         'help': 'domain file'}),
    (('-l', '--n_learning_epoch'),
        {'metavar': 'NUM_LEARNING_EPOCHS',
         'dest': 'n_learning_epoch',
         'default': 0,
         'type': int,
         'help': 'number of learning epochs'}),
    (('-i', '--n_inference_epoch'),
        {'metavar': 'NUM_INFERENCE_EPOCHS',
         'dest': 'n_inference_epoch',
         'default': 0,
         'type': int,
         'help': 'number of inference epochs'}),
    (('-s', '--stepsize', '-a', '--alpha'),
        {'metavar': 'LEARNING_STEPSIZE',
         'dest': 'stepsize',
         'default': 0.01,
         'type': float,
         'help': 'stepsize for learning'}),
    (('-d', '--decay', '--diminish'),
        {'metavar': 'LEARNING_DECAY',
         'dest': 'decay',
         'default': 0.95,
         'type': float,
         'help': 'decay for updating stepsize during learning'}),
    (('-r', '--reg_param'),
        {'metavar': 'LEARNING_REGULARIZATION_PARAM',
         'dest': 'reg_param',
         'default': 0.01,
         'type': float,
         'help': 'regularization penalty'}),
    (tuple(['--regularization']),
        {'metavar': 'REGULARIZATION',
         'dest': 'regularization',
         'default': 2,
         'type': int,
         'help': 'regularization (l1 or l2) [Enter as "1" or "2"]'}),
    (('-k', '--truncation'),
        {'metavar': 'TRUNCATION',
         'dest': 'truncation',
         'default': 1,
         'type': int,
         'help': 'If using l1 regularization, truncation is applied with '
                 'probability 1/k and with magnitude '
                 'step_size * reg_param * k. If not using l1 regularization, '
                 'this parameter has no effect.'}),
    (('-b', '--burn_in'),
        {'metavar': 'BURN_IN',
         'dest': 'burn_in',
         'default': 0,
         'type': int,
         'help': 'number of burn-in epochs'}),
    (('-t', '--threads', '--n_threads'),
        {'metavar': 'NUM_THREADS',
         'dest': 'nthreads',
         'default': 1,
         'type': int,
         'help': 'number of threads to be used'}),
    (('-u', '--dburl'),
        {'metavar': 'DATABASE_URL',
         'dest': 'dburl',
         'default': '',
         'type': str,
         'help': 'url to database holding factor graph information'})
]

flags = [
    (tuple(['--sample_evidence']),
        {'default': True,
         'dest': 'sample_evidence',
         'action': 'store_true',
         'help': 'sample evidence variables'}),
    (tuple(['--learn_non_evidence']),
        {'default': False,
         'dest': 'learn_non_evidence',
         'action': 'store_true',
         'help': 'learn from non-evidence variables'}),
    (('-q', '--quiet'),
        {'default': False,
         'dest': 'quiet',
         'action': 'store_true',
         'help': 'quiet'}),
    (tuple(['--verbose']),
        {'default': False,
         'dest': 'verbose',
         'action': 'store_true',
         'help': 'verbose'})
]


class NumbSkull(object):
    """TODO: Main class for numbskull."""

    def __init__(self, **kwargs):
        """TODO.

        Parameters
        ----------
        paramater : type
           This is a parameter

        Returns
        -------
        describe : type
            Expanation
        """
        # Initialize default execution arguments
        arg_defaults = {}
        for arg, opts in arguments:
            if 'directory' in arg[0]:
                arg_defaults['directory'] = opts['default']
            else:
                arg_defaults[opts['dest']] = opts['default']
        # Initialize default execution flags
        for arg, opts in flags:
            arg_defaults[opts['dest']] = opts['default']
        for (arg, default) in arg_defaults.items():
            setattr(self, arg, kwargs.get(arg, default))

        self.factorGraphs = []

    def loadFactorGraphRaw(self, weight, variable, factor, fmap,
                           vmap, factor_index,
                           var_copies=1, weight_copies=1):
        """TODO."""
        fg = FactorGraph(weight, variable, factor, fmap, vmap, factor_index,
                         var_copies, weight_copies,
                         len(self.factorGraphs), self.nthreads)
        self.factorGraphs.append(fg)

    def loadFactorGraph(self, weight, variable, factor, fmap, domain_mask,
                        edges, var_copies=1, weight_copies=1,
                        factors_to_skip=np.empty(0, np.int64)):
        """TODO."""
        # Note: factors_to_skip must be sorted
        # Assert input arguments correspond to NUMPY arrays
        assert(type(weight) == np.ndarray and weight.dtype == Weight)
        assert(type(variable) == np.ndarray and variable.dtype == Variable)
        assert(type(factor) == np.ndarray and factor.dtype == Factor)
        assert(type(fmap) == np.ndarray and fmap.dtype == FactorToVar)
        assert(type(domain_mask) == np.ndarray and
               domain_mask.dtype == np.bool)
        assert(type(edges) == int or
               type(edges) == long or
               type(edges) == np.int64)
        assert(type(factors_to_skip) == np.ndarray and
               factors_to_skip.dtype == np.int64)

        # Initialize metadata
        meta = {}
        meta['weights'] = weight.shape[0]
        meta['variables'] = variable.shape[0]
        meta['factors'] = factor.shape[0]
        # TODO: should probably just delete edges as an argument
        # Not really needed (can just be computed from factors)
        edges = sum(factor["arity"]) - sum(factor[factors_to_skip]["arity"])
        meta['edges'] = edges

        # count total number of VTF records needed
        num_vtfs = 0
        for var in variable:
            var["vtf_offset"] = num_vtfs
            if var["dataType"] == 0:  # boolean
                num_vtfs += 1
            else:
                num_vtfs += var["cardinality"]

        vmap = np.zeros(num_vtfs, VarToFactor)
        # factors_to_skip is a list of indices of factors
        # these factors need to exist for the distributed sampler
        # but cannot be sampled
        # TODO: edges is really poorly defined with factors_to_skip
        factor_index = np.zeros(meta["edges"], np.int64)

        # Numba-based method. Defined in dataloading.py
        compute_var_map(variable, factor, fmap, vmap,
                        factor_index, domain_mask, factors_to_skip)

        fg = FactorGraph(weight, variable, factor, fmap, vmap, factor_index,
                         var_copies, weight_copies,
                         len(self.factorGraphs), self.nthreads)
        self.factorGraphs.append(fg)

    def loadFGFromFile(self, directory=None, metafile=None, weightfile=None,
                       variablefile=None, factorfile=None, domainfile=None,
                       var_copies=1, weight_copies=1):
        """TODO."""
        # init necessary input arguments
        if not self.directory:
            print("No factor graph specified")
            return
        else:
            directory = self.directory

        metafile = self.metafile if not metafile else metafile
        weightfile = self.weightfile if not weightfile else weightfile
        variablefile = self.variablefile if not variablefile else variablefile
        factorfile = self.factorfile if not factorfile else factorfile
        domainfile = self.domainfile if not domainfile else domainfile
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
        weight = np.zeros(meta["weights"], Weight)

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
        variable = np.zeros(meta["variables"], Variable)

        # NUMBA-based method. Defined in dataloading.py
        load_variables(variable_data, meta["variables"], variable)
        sys.stdout.flush()
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

        # count total number of VTF records needed
        num_vtfs = 0
        for var in variable:
            var["vtf_offset"] = num_vtfs
            if var["dataType"] == 0:  # boolean
                num_vtfs += 1
            else:
                num_vtfs += var["cardinality"]
        print("#VTF = %s" % num_vtfs)
        sys.stdout.flush()

        # generate variable-to-factor map
        vmap = np.zeros(num_vtfs, VarToFactor)
        factor_index = np.zeros(meta["edges"], np.int64)

        # load domains
        # whether a var has domain spec
        domain_mask = np.zeros(meta["variables"], np.bool)
        domain_file = directory + "/" + domainfile
        if os.path.isfile(domain_file) and os.stat(domain_file).st_size > 0:
            domain_data = np.memmap(directory + "/" + domainfile, mode="c")
            load_domains(domain_data, domain_mask, vmap, variable)
            sys.stdout.flush()

        # load factors
        factor_data = np.memmap(directory + "/" + factorfile, mode="c")
        factor = np.zeros(meta["factors"], Factor)
        fmap = np.zeros(meta["edges"], FactorToVar)

        # Numba-based method. Defined in dataloading.py
        load_factors(factor_data, meta["factors"],
                     factor, fmap, domain_mask, variable, vmap)
        sys.stdout.flush()

        # Numba-based method. Defined in dataloading.py
        compute_var_map(variable, factor, fmap, vmap,
                        factor_index, domain_mask)
        print("COMPLETED VMAP INDEXING")
        sys.stdout.flush()

        fg = FactorGraph(weight, variable, factor, fmap, vmap, factor_index,
                         var_copies, weight_copies,
                         len(self.factorGraphs), self.nthreads)
        self.factorGraphs.append(fg)

    def getFactorGraph(self, fgID=0):
        """TODO."""
        return self.factorGraphs[fgID]

    def inference(self, fgID=0, out=True):
        """TODO."""
        burn_in = self.burn_in
        n_inference_epoch = self.n_inference_epoch

        self.factorGraphs[fgID].inference(burn_in, n_inference_epoch,
                                          sample_evidence=self.sample_evidence,
                                          diagnostics=not self.quiet)
        if out:
            output_file = os.path.join(
                self.output_dir, "inference_result.out.text")
            self.factorGraphs[fgID].dump_probabilities(output_file,
                                                       n_inference_epoch)

    def learning(self, fgID=0, out=True):
        """TODO."""
        burn_in = self.burn_in
        n_learning_epoch = self.n_learning_epoch
        stepsize = self.stepsize
        decay = self.decay
        regularization = self.regularization
        reg_param = self.reg_param
        truncation = self.truncation
        fg = self.factorGraphs[fgID]
        fg.learn(burn_in, n_learning_epoch,
                 stepsize, decay, regularization, reg_param, truncation,
                 diagnostics=not self.quiet,
                 verbose=self.verbose,
                 learn_non_evidence=self.learn_non_evidence)
        if out:
            output_file = os.path.join(
                self.output_dir, "inference_result.out.weights.text")
            self.factorGraphs[fgID].dump_weights(output_file)


def load(argv=None):
    """TODO."""
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


def main(argv=None):
    """Duh."""
    ns = load(argv)
    ns.learning()
    ns.inference()
