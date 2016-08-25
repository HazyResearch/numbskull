from __future__ import print_function
import numba
from numba import jit
import numpy as np
import math
import random
from inference import draw_sample, eval_factor

@jit(nopython=True,cache=True,nogil=True)
def learnthread(shardID, nshards, step, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value, learn_non_evidence):
    # Identify start and end variable
    nvar  = variable.shape[0]
    start = ((nvar / nshards) + 1) * shardID
    end   = min(((nvar / nshards) +1) * (shardID + 1), nvar)

    for var_samp in range(start,end):
        # If variable is an observation do not do anything
        if variable[var_samp]["isEvidence"] == 2:
            pass
        else:
            sample_and_sgd(var_samp, step, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value, learn_non_evidence)

@jit(nopython=True,cache=True,nogil=True)
def sample_and_sgd(var_samp, step, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value, learn_non_evidence):

    # If learn_non_evidence then store the previous value in a tmp variable
    # (corresponds to value with weights from previous iteration) 
    # then sample and compute the gradient.
    if learn_non_evidence:
       evidence = var_value[var_copy][var_samp] 
    # If evidence then store the initial value in a tmp variable
    # then sample and compute the gradient.
    else:
       evidence = variable[var_samp]["initialValue"]

    # Sample the variable
    proposal = draw_sample(var_samp, var_copy, weight_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value)

    var_value[var_copy][var_samp] = proposal

    if not learn_non_evidence and variable[var_sample]["isEvidence"] != 1:
        return

    # Compute the gradient and update the weights
    # Iterate over corresponding factors
    for i in range(vstart[var_samp], vstart[var_samp + 1]):
        factor_id = vmap[i]
        weight_id = factor[factor_id]["weightId"]

        if weight[weight_id]["isFixed"]:
            continue
        # Boolean variable
        if variable[var_samp]["dataType"] == 0:
        # Gradient for SGD
        p0 = eval_factor(factor_id, var_samp, evidence, var_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value)
        p1 = eval_factor(factor_id, var_samp, proposal, var_copy, weight, variable, factor, fstart, fmap, vstart, vmap, equalPred, Z, count, var_value, weight_value)
        gradient = p1 - p0
   
    # TODO: return none or sampled var?
    # Sample the variable
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



