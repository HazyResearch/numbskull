"""TODO."""

from __future__ import print_function
import numba
from numba import jit
import numpy as np
import math
import random
from inference import draw_sample, eval_factor


@jit(nopython=True, cache=True, nogil=True)
def learnthread(shardID, nshards, step, regularization, reg_param,
                var_copy, weight_copy, weight,
                variable, factor, fmap,
                vmap, factor_index, Z, var_value, var_value_evid,
                weight_value, learn_non_evidence):
    """TODO."""
    # Identify start and end variable
    nvar = variable.shape[0]
    start = ((nvar / nshards) + 1) * shardID
    end = min(((nvar / nshards) + 1) * (shardID + 1), nvar)
    Z = Z.copy()
    for var_samp in range(start, end):
        sample_and_sgd(var_samp, step, regularization, reg_param,
                       var_copy, weight_copy, weight, variable,
                       factor, fmap, vmap,
                       factor_index, Z, var_value, var_value_evid,
                       weight_value, learn_non_evidence)


@jit(nopython=True, cache=True, nogil=True)
def get_factor_id_range(variable, vmap, var_samp, val):
    """TODO."""
    varval_off = val
    if variable[var_samp]["dataType"] == 0:
        varval_off = 0
    vtf = vmap[variable[var_samp]["vtf_offset"] + varval_off]
    start = vtf["factor_index_offset"]
    end = start + vtf["factor_index_length"]
    return (start, end)


@jit(nopython=True, cache=True, nogil=True)
def sample_and_sgd(var_samp, step, regularization, reg_param, var_copy,
                   weight_copy, weight, variable, factor, fmap,
                   vmap, factor_index, Z, var_value, var_value_evid,
                   weight_value, learn_non_evidence):
    """TODO."""
    # If learn_non_evidence sample twice.
    # The method corresponds to expectation-conjugate descent.
    if variable[var_samp]["isEvidence"] != 1:
        evidence = draw_sample(var_samp, var_copy, weight_copy,
                               weight, variable, factor,
                               fmap, vmap, factor_index, Z,
                               var_value_evid, weight_value)
    # If evidence then store the initial value in a tmp variable
    # then sample and compute the gradient.
    else:
        evidence = variable[var_samp]["initialValue"]
    var_value_evid[var_copy][var_samp] = evidence
    # Sample the variable
    proposal = draw_sample(var_samp, var_copy, weight_copy, weight,
                           variable, factor, fmap, vmap,
                           factor_index, Z, var_value, weight_value)

    var_value[var_copy][var_samp] = proposal
    if not learn_non_evidence and variable[var_samp]["isEvidence"] != 1:
        return
    # Compute the gradient and update the weights
    # Iterate over corresponding factors

    range_fids = get_factor_id_range(variable, vmap, var_samp, evidence)
    if evidence != proposal:
        range_prop = get_factor_id_range(variable, vmap, var_samp, proposal)
        fids = np.concatenate((factor_index[range_fids[0]:range_fids[1]],
                               factor_index[range_prop[0]:range_prop[1]]))
        fids.sort()
    else:
        fids = factor_index[range_fids[0]:range_fids[1]]

    # go over all factor ids, ignoring dupes
    last_fid = -1  # numba 0.28 would complain if this were None
    for factor_id in fids:
        if factor_id == last_fid:
            continue
        last_fid = factor_id
        weight_id = factor[factor_id]["weightId"]
        if weight[weight_id]["isFixed"]:
            continue
        # Compute Gradient
        p0 = eval_factor(factor_id, var_samp,
                         evidence, var_copy,
                         variable, factor, fmap,
                         var_value_evid) * factor[factor_id]["featureValue"]
        p1 = eval_factor(factor_id, var_samp,
                         proposal, var_copy,
                         variable, factor, fmap,
                         var_value) * factor[factor_id]["featureValue"]
        gradient = p1 - p0
        # Update weight
        w = weight_value[weight_copy][weight_id]
        if regularization == 2:
            w *= (1.0 / (1.0 + reg_param * step))
            w -= step * gradient
        elif regularization == 1:
            # Truncated Gradient
            # "Sparse Online Learning via Truncated Gradient"
            #  Langford et al. 2009
            l1delta = reg_param * step
            w -= step * gradient
            w = max(0, w - l1delta) if w > 0 \
                else min(0, w + l1delta)
        else:
            w -= step * gradient
        weight_value[weight_copy][weight_id] = w
