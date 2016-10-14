#!/usr/bin/env python

"""TODO."""

from __future__ import print_function
import numbskull
from numbskull.numbskulltypes import *
import numpy as np


def factor(f, args):
    """THIS IS A DOCSTRING."""
    if f == FUNC_IMPLY_NATURAL:
        # TODO
        pass
    elif f == FUNC_OR:
        return 1 if any(args) else -1
    elif f == FUNC_EQUAL:
        # TODO
        pass
    elif f == FUNC_AND or FUNC_ISTRUE:
        return 1 if all(args) else -1
    elif f == FUNC_LINEAR:
        # TODO
        pass
    elif f == FUNC_RATIO:
        # TODO
        pass
    elif f == FUNC_LOGICAL:
        # TODO
        pass
    elif f == FUNC_IMPLY_MLN:
        # TODO
        pass
    else:
        raise NotImplemented("FACTOR " + str(f) + " not implemented.")


for (key, value) in numbskull.inference.FACTORS.items():

    print(key)

    variables = 2
    if key == "DP_GEN_DEP_FIXING" or key == "DP_GEN_DEP_REINFORCING":
        # These factor functions requires three vars to work
        variables = 3
    edges = variables

    weight = np.zeros(1, Weight)
    variable = np.zeros(variables, Variable)
    factor = np.zeros(1, Factor)
    fmap = np.zeros(edges, FactorToVar)
    domain_mask = np.zeros(variables, np.bool)

    weight[0]["isFixed"] = True
    weight[0]["initialValue"] = 1

    for i in range(variables):
        variable[i]["isEvidence"] = 0
        variable[i]["initialValue"] = 0
        variable[i]["dataType"] = 0
        variable[i]["cardinality"] = 2

    factor[0]["factorFunction"] = value
    factor[0]["weightId"] = 0
    factor[0]["featureValue"] = 1
    factor[0]["arity"] = variables
    factor[0]["ftv_offset"] = 0

    for i in range(variables):
        fmap[i]["vid"] = i

    ns = numbskull.NumbSkull(n_inference_epoch=100,
                             n_learning_epoch=100,
                             quiet=True)

    ns.loadFactorGraph(weight, variable, factor, fmap, domain_mask, edges)

    ns.learning()
    ns.inference()
    print(ns.factorGraphs[0].count)
