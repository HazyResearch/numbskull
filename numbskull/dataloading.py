from __future__ import print_function
import numba
from numba import jit
import numpy as np


# HELPER METHODS #
def dataType(i):
    return {0: "Boolean",
            1: "Categorical"}.get(i, "Unknown")


# DEFINE NUMBA-BASED DATA LOADING HELPER METHODS #
@jit(nopython=True, cache=True)
def compute_var_map(fstart, fmap, vstart, vmap):
    for i in fmap:
        vstart[i + 1] += 1

    for i in range(len(vstart) - 1):
        vstart[i + 1] += vstart[i]
    index = vstart.copy()

    for i in range(len(fstart) - 1):
        for j in range(fstart[i], fstart[i + 1]):
            vmap[index[fmap[j]]] = i
            index[fmap[j]] += 1


@jit(nopython=True, cache=True)
def reverse(data, start, end):
    end -= 1
    while (start < end):
        data[start], data[end] = data[end], data[start]
        start += 1
        end -= 1


# DEFINE NUMBA-BASED DATA LOADING METHODS #
@jit(nopython=True, cache=True)
def load_weights(data, nweights, weights):
    for i in range(nweights):
        # TODO: read types from struct?
        # TODO: byteswap only if system is little-endian

        reverse(data, 17 * i, 17 * i + 8)
        weightId = np.frombuffer(data[(17 * i):(17 * i + 8)], dtype=np.int64)[0]
        isFixed = data[17 * i + 8]
        reverse(data, 17 * i + 9, 17 * i + 17)
        initialValue = np.frombuffer(data[(17 * i + 9):(17 * i + 17)], dtype=np.float64)[0]

        weights[weightId]["isFixed"] = isFixed
        weights[weightId]["initialValue"] = initialValue
    print("DONE WITH WEIGHTS")


@jit(nopython=True, cache=True)
def load_variables(data, nvariables, variables):
    for i in range(nvariables):
        # TODO: read types from struct?
        # TODO: byteswap only if system is little-endian

        reverse(data, 19 * i, 19 * i + 8)
        variableId = np.frombuffer(data[(19 * i):(19 * i + 8)], dtype=np.int64)[0]
        isEvidence = data[19 * i + 8]
        reverse(data, 19 * i + 9, 19 * i + 13)
        initialValue = np.frombuffer(data[(19 * i + 9):(19 * i + 13)], dtype=np.int32)[0]
        reverse(data, 19 * i + 13, 19 * i + 15)
        dataType = np.frombuffer(data[(19 * i + 13):(19 * i + 15)], dtype=np.int16)[0]
        reverse(data, 19 * i + 15, 19 * i + 19)
        cardinality = np.frombuffer(data[(19 * i + 15):(19 * i + 19)], dtype=np.int32)[0]

        variables[variableId]["isEvidence"] = isEvidence
        variables[variableId]["initialValue"] = initialValue
        variables[variableId]["dataType"] = dataType
        variables[variableId]["cardinality"] = cardinality
    print("DONE WITH VARS")


@jit(nopython=True, cache=True)
def load_factors(data, nfactors, factors, fstart, fmap, equalPredicate):
    index = 0
    for i in range(nfactors):
        reverse(data, index, index + 2)
        factors[i]["factorFunction"] = np.frombuffer(data[index:(index + 2)], dtype=np.int16)[0]

        reverse(data, index + 2, index + 6)
        arity = np.frombuffer(data[(index + 2):(index + 6)], dtype=np.int32)[0]

        index += 6  # TODO: update index once per loop?

        fstart[i + 1] = fstart[i] + arity
        for j in range(arity):
            reverse(data, index, index + 8)
            fmap[fstart[i] + j] = np.frombuffer(data[index:(index + 8)], dtype=np.int64)[0]
            reverse(data, index + 8, index + 12)
            equalPredicate[fstart[i] + j] = np.frombuffer(data[(index + 8):(index + 12)], dtype=np.int32)[0]
            index += 12

        # TODO: handle FUNC_AND_CATEGORICAL
        reverse(data, index, index + 8)
        factors[i]["weightId"] = np.frombuffer(data[index:(index + 8)], dtype=np.int64)[0]
        reverse(data, index + 8, index + 16)
        factors[i]["featureValue"] = np.frombuffer(data[(index + 8):(index + 16)], dtype=np.float64)[0]
        index += 16
    print("DONE WITH FACTORS")
