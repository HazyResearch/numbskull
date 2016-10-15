#!/usr/bin/env python

"""This tests learning for labelling functions."""

from __future__ import print_function, absolute_import
import numpy as np
import numbskull
from numbskull.numbskulltypes import *
import math


def index_to_values(index, num_lf):
    value = [0] * (1 + num_lf)
    value[0] = index % 2
    index = index // 2
    for i in range(num_lf):
        value[i + 1] = index % 3
        index = index // 3
    return value

def create_fg(prior, accuracy, abstain, copies):
    """
    This creates copies of the following factor graph.
                 istrue (weight = prior)
                              |
                             y_i
                             /|\
                            / | \
                           /  |  \
                          /   |   \
                         /    |    \
                    LF_{i1}  ...  LF_{in}
                ( weight =  )   ( weight =  )
                 accuracy[1]     accuracy[n]

    Arguments:
        prior: one floating-point value
        weight: list of floating-point values
        abstain: list of floating-point values (same number as weight)
        copies: integer
    Returns:
        list of arguments that can be passed to numbskull.loadFactorGraph
    """

    weights = 1 + len(accuracy)
    variables = copies * (1 + len(accuracy))
    factors = copies * (1 + len(accuracy))
    edges = copies * (1 + 2 * len(accuracy))

    weight = np.zeros(weights, Weight)
    variable = np.zeros(variables, Variable)
    factor = np.zeros(factors, Factor)
    fmap = np.zeros(edges, FactorToVar)
    domain_mask = np.zeros(variables, np.bool)

    states = 2 * 3 ** len(accuracy)
    Z = np.zeros(states, np.float64)
    for i in range(states):
        value = index_to_values(i, len(accuracy))

        y = value[0]
        lfs = value[1:]

        Z[i] = prior * (2 * y - 1)
        for (j, lf) in enumerate(lfs):
            lf = lf - 1 # remap to standard -1, 0, 1
            if lf != 0:
                Z[i] += accuracy[j] * lf * y

        Z[i] = math.exp(Z[i])
    Z = np.cumsum(Z)
    Z = Z / Z[-1]

    for w in weight:
        w["isFixed"] = False
        w["initialValue"] = 0


    for copy in range(copies):
        r = np.random.rand()
        index = np.argmax(Z >= r)
        value = index_to_values(index, len(accuracy))
        y = value[0]
        lf = value[1:]

        # y variable
        variable[copy * (1 + len(accuracy))]["isEvidence"] = 0  # query
        variable[copy * (1 + len(accuracy))]["initialValue"] = y
        variable[copy * (1 + len(accuracy))]["dataType"] = 0  # binary
        variable[copy * (1 + len(accuracy))]["cardinality"] = 2

        # labelling function variable
        for i in range(len(accuracy)):
            variable[copy * (1 + len(accuracy)) + 1 + i]["isEvidence"] = 1  # evidence
            variable[copy * (1 + len(accuracy)) + 1 + i]["initialValue"] = lf[i]
            variable[copy * (1 + len(accuracy)) + 1 + i]["dataType"] = 1  # categorical
            variable[copy * (1 + len(accuracy)) + 1 + i]["cardinality"] = 3

        # Class prior
        factor[copy * (1 + len(accuracy))]["factorFunction"] = 18  # FUNC_DP_GEN_CLASS_PRIOR
        factor[copy * (1 + len(accuracy))]["weightId"] = 0
        factor[copy * (1 + len(accuracy))]["featureValue"] = 1
        factor[copy * (1 + len(accuracy))]["arity"] = 1
        factor[copy * (1 + len(accuracy))]["ftv_offset"] = copy * (1 + len(accuracy))
        fmap  [copy * (1 + 2 * len(accuracy))]["vid"] = copy * (1 + len(accuracy))

        # Labelling function accuracy
        for i in range(len(accuracy)):
            factor[copy * (1 + len(accuracy)) + 1 + i]["factorFunction"] = 21  # FUNC_DP_GEN_LF_ACCURACY
            factor[copy * (1 + len(accuracy)) + 1 + i]["weightId"] = i
            factor[copy * (1 + len(accuracy)) + 1 + i]["featureValue"] = 1
            factor[copy * (1 + len(accuracy)) + 1 + i]["arity"] = 2
            factor[copy * (1 + len(accuracy)) + 1 + i]["ftv_offset"] = copy * (1 + len(accuracy)) + 1 + 2 * i
            fmap  [copy * (1 + 2 * len(accuracy)) + 1 + 2 * i]["vid"] = copy * (1 + len(accuracy))      # y
            fmap  [copy * (1 + 2 * len(accuracy)) + 2 + 2 * i]["vid"] = copy * (1 + len(accuracy)) + 1  # labeling func i

    return weight, variable, factor, fmap, domain_mask, edges

ns = numbskull.NumbSkull(n_inference_epoch=100,
                         n_learning_epoch=500,
                         quiet=True,
                         learn_non_evidence=True,
                         stepsize=0.1,
                         burn_in=100,
                         decay=0.95,
                         reg_param=0.1)

prior = 0
accuracy = [0, 0, 0]
abstain = [0, 0, 0]
copies = 3
fg = create_fg(prior, accuracy, abstain, copies)
ns.loadFactorGraph(*fg)
ns.learning()
print(ns.factorGraphs[0].weight_value)
