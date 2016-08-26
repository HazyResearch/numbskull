#!/usr/bin/env python

from numbskull import numbskull
from numbskull.numbskulltypes import *
import numpy as np

weight = np.empty(1, Weight)
variable = np.empty(2, Variable)
factor = np.empty(1, Factor)
equalPredicate = np.empty(1, np.int32)
fstart = np.empty(1, np.int64)
fmap = np.empty(1, np.int64)
edges = 2

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

factor[0]["factorFunction"] = 1
factor[0]["weightId"] = 0
factor[0]["featureValue"] = 0

ns = numbskull.NumbSkull(n_inference_epoch=100, n_learning_epoch=100)
ns.loadFactorGraph(weight, variable, factor, fstart, fmap, equalPredicate, edges)

ns.learning()
ns.inference()
print(ns.factorGraphs[0].count)
