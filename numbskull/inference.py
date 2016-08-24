from __future__ import print_function
import numba
from numba import jit
import numpy as np
import math

@jit(nopython=True,cache=True,nogil=True)
def gibbsthread(shardID, nshards, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value, burnin):
    # Indentify start and end variable
    nvar  = variable.shape[0]
    start = ((nvar / nshards) + 1) * shardID
    end   = min(((nvar / nshards) +1) * (shardID + 1), nvar)
    # TODO: give option do not store result, or just store tally
    for var_samp in range(start,end):
        # TODO: sample evidence
        # TODO: observation
        if variable[var_samp]["isEvidence"] == 0:
            v = draw_sample(var_samp, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value)
            var_value[var_copy][var_samp] = v
            if not burnin:
                count[var_samp] += v

    var_value[var_copy][var_samp] = draw_sample(var_samp, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value)
    return var_value[var_copy][var_samp]


@jit(nopython=True,cache=True,nogil=True)
def draw_sample(var_samp, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value):
    cardinality = variable[var_samp]["cardinality"]
    for value in range(cardinality):
        Z[value] = np.exp(potential(var_samp, value, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value))

    for j in range(1, cardinality):
        Z[j] += Z[j - 1]

    z = np.random.rand() * Z[cardinality - 1]
    # TODO: I think this looks at the full vector, will be slow if one var has high cardinality
    return np.argmax(Z >= z)


@jit(nopython=True,cache=True,nogil=True)
def potential(var_samp, value, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value):
    p = 0.0
    for k in range(vstart[var_samp], vstart[var_samp + 1]):
        factor_id = vmap[k]
        p += weight_value[weight_copy][factor[vmap[k]]["weightId"]]*eval_factor(factor_id, var_samp, value, var_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value)
    return p


@jit(nopython=True,cache=True,nogil=True)
def eval_factor(factor_id, var_samp, value, var_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value):
    if factor[factor_id]["factorFunction"] == 3: # FUNC_EQUAL
        v = value if (fmap[fstart[factor_id]] == var_samp) else var_value[var_copy][fmap[fstart[factor_id]]]
        for l in range(fstart[factor_id] + 1, fstart[factor_id + 1]):
            w = value if (fmap[l] == var_samp) else var_value[var_copy][fmap[l]]
            if v != w:
                return -1
        return 1
    elif factor[factor_id]["factorFunction"] == 4: # FUNC_ISTRUE
        for l in range(fstart[factor_id], fstart[factor_id + 1]):
            v = value if (fmap[l] == var_samp) else var_value[var_copy][fmap[l]]
            if v == 0:
                return -1
        return 1
    else: # FUNC_UNDEFINED
        print("Error: Factor Function", factor[factor_id]["factorFunction"], "( used in factor", factor_id, ") is not implemented.")
        raise NotImplementedError("Factor function is not implemented.")

