#!/usr/bin/env python

import numbskull
from numbskull.numbskulltypes import *
import numpy as np

for (key, value) in numbskull.inference.FACTORS.iteritems():

    variables = 2

    weight = np.empty(1, Weight)
    variable = np.empty(variables, Variable)
    factor = np.empty(1, Factor)
    equalPredicate = np.empty(variables, np.int32)  # TODO: for binary?
    fstart = np.empty(2, np.int64)
    fmap = np.empty(variables, np.int64)

    edges = variables

    weight[0]["isFixed"] = True
    weight[0]["initialValue"] = 1

    variable[0]["isEvidence"] = 0
    variable[0]["initialValue"] = 0
    variable[0]["dataType"] = 0
    variable[0]["cardinality"] = 2

    variable[1]["isEvidence"] = 0
    variable[1]["initialValue"] = 0
    variable[1]["dataType"] = 0
    variable[1]["cardinality"] = 2

    factor[0]["factorFunction"] = value
    factor[0]["weightId"] = 0
    factor[0]["featureValue"] = 0

    fstart[0] = 0
    fstart[1] = variables

    for i in range(variables):
        fmap[i] = i

    ns = numbskull.NumbSkull(n_inference_epoch=100,
                             n_learning_epoch=100,
                             quiet=True)
    ns.loadFactorGraph(weight, variable, factor, fstart,
                       fmap, equalPredicate, edges)

    ns.learning()
    ns.inference()
    print(ns.factorGraphs[0].count)
