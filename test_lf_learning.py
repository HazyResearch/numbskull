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

    n = len(accuracy)  # number of labelling functions

    weights = 1 + n
    variables = copies * (1 + n)
    factors = copies * (1 + n)
    edges = copies * (1 + 2 * n)

    weight = np.zeros(weights, Weight)
    variable = np.zeros(variables, Variable)
    factor = np.zeros(factors, Factor)
    fmap = np.zeros(edges, FactorToVar)
    domain_mask = np.zeros(variables, np.bool)

    states = 2 * 3 ** n
    Z = np.zeros(states, np.float64)
    for i in range(states):
        value = index_to_values(i, n)

        y = value[0]
        lfs = value[1:]

        Z[i] = prior * (2 * y - 1)
        for (j, lf) in enumerate(lfs):
            lf = lf - 1  # remap to standard -1, 0, 1
            if lf != 0:
                Z[i] += accuracy[j] * lf * (2 * y - 1)
        # TODO: abstain not handled yet

        Z[i] = math.exp(Z[i])

    Z = np.cumsum(Z)
    Z = Z / Z[-1]
    print(Z)

    for w in weight:
        w["isFixed"] = False
        w["initialValue"] = 1.0
    weight[0]["initialValue"] = 0

    for copy in range(copies):
        r = np.random.rand()
        index = np.argmax(Z >= r)
        value = index_to_values(index, n)
        y = value[0]
        lf = value[1:]

        # y variable
        variable[copy * (1 + n)]["isEvidence"] = 0  # query
        variable[copy * (1 + n)]["initialValue"] = 0  # Do not actually show y
        variable[copy * (1 + n)]["dataType"] = 0  # not sparse
        variable[copy * (1 + n)]["cardinality"] = 2

        # labelling function variable
        for i in range(n):
            variable[copy * (1 + n) + 1 + i]["isEvidence"] = 1  # evidence
            variable[copy * (1 + n) + 1 + i]["initialValue"] = lf[i]
            variable[copy * (1 + n) + 1 + i]["dataType"] = 0  # not sparse
            variable[copy * (1 + n) + 1 + i]["cardinality"] = 3

        # Class prior
        factor[copy * (1 + n)]["factorFunction"] = 18  # DP_GEN_CLASS_PRIOR
        factor[copy * (1 + n)]["weightId"] = 0
        factor[copy * (1 + n)]["featureValue"] = 1
        factor[copy * (1 + n)]["arity"] = 1
        factor[copy * (1 + n)]["ftv_offset"] = copy * (1 + 2 * n)
        fmap[copy * (1 + 2 * n)]["vid"] = copy * (1 + n)

        # Labelling function accuracy
        for i in range(n):
            factor_index = copy * (1 + n) + 1 + i
            factor[factor_index]["factorFunction"] = 21  # DP_GEN_LF_ACCURACY
            factor[factor_index]["weightId"] = i + 1
            factor[factor_index]["featureValue"] = 1
            factor[factor_index]["arity"] = 2
            factor[factor_index]["ftv_offset"] = copy * (1 + 2 * n) + 1 + 2 * i

            fmap_index = copy * (1 + 2 * n) + 1 + 2 * i
            fmap[fmap_index]["vid"] = copy * (1 + n)  # y
            fmap[fmap_index + 1]["vid"] = copy * (1 + n) + i + 1  # LF i

    return weight, variable, factor, fmap, domain_mask, edges

learn = 100
ns = numbskull.NumbSkull(n_inference_epoch=100,
                         n_learning_epoch=learn,
                         quiet=True,
                         learn_non_evidence=True,
                         stepsize=0.0001,
                         burn_in=100,
                         decay=0.001 ** (1.0 / learn),
                         regularization=1,
                         reg_param=0.01)

prior = 0
accuracy = [1, 0.5]
abstain = [0, 0, 0]
copies = 10

fg = create_fg(prior, accuracy, abstain, copies)
print("weight")
print(fg[0])
print()

print("variable")
print(fg[1])
print()

print("factor")
print(fg[2])
print()

print("fmap")
print(fg[3])
print()

print("domain_mask")
print(fg[4])
print()

print("edges")
print(fg[5])
print()

ns.loadFactorGraph(*fg)
print(ns.factorGraphs[0].weight_value)
ns.learning()
print(ns.factorGraphs[0].weight_value)
