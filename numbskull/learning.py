from __future__ import print_function
import numba
from numba import jit
import numpy as np
import math
import random
from inference import draw_sample, eval_factor


@jit(nopython=True, cache=True, nogil=True)
def learnthread(shardID, nshards, step, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value):
    # Identify start and end variable
    nvar = variable.shape[0]
    start = ((nvar / nshards) + 1) * shardID
    end = min(((nvar / nshards) + 1) * (shardID + 1), nvar)

    for var_samp in range(start, end):
        if variable[var_samp]["isEvidence"] == 2:
            pass
        else:
            sample_and_sgd(var_samp, step, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value)


@jit(nopython=True, cache=True, nogil=True)
def sample_and_sgd(var_samp, step, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value):
    # TODO: return none or sampled var?
    # TODO: return if is observation
    var_value[var_copy][var_samp] = draw_sample(var_samp, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value)

    # TODO: set initialValue
    # TODO: if isevidence or learn_non_evidence
    if variable[var_samp]["isEvidence"] == 1:
        for i in range(vstart[var_samp], vstart[var_samp + 1]):
            factor_id = vmap[i]
            weight_id = factor[factor_id]["weightId"]

            if not weight[weight_id]["isFixed"]:
                # TODO: save time by checking if initialValue and value are equal first?
                p0 = eval_factor(factor_id, var_samp, variable[var_samp]["initialValue"], var_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value)
                p1 = eval_factor(factor_id, var_samp, var_value[var_copy][var_samp], var_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value)
                weight_value[weight_copy][weight_id] += step * (p0 - p1)
