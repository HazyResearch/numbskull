from __future__ import print_function
import numba
from numba import jit
import numpy as np
import math


@jit(nopython=True, cache=True, nogil=True)
def gibbsthread(shardID, nshards, var_copy, weight_copy, weight, variable,
                factor, fstart, fmap, vstart, vmap, equalPred, Z, cstart,
                count, var_value, weight_value, burnin):
    # Indentify start and end variable
    nvar = variable.shape[0]
    start = ((nvar / nshards) + 1) * shardID
    end = min(((nvar / nshards) + 1) * (shardID + 1), nvar)
    # TODO: give option do not store result, or just store tally
    for var_samp in range(start, end):
        # TODO: sample evidence
        # TODO: observation
        if variable[var_samp]["isEvidence"] == 0:
            v = draw_sample(var_samp, var_copy, weight_copy, weight, variable,
                            factor, fstart, fmap, vstart, vmap, equalPred, Z,
                            count, var_value, weight_value)
            var_value[var_copy][var_samp] = v
            if not burnin:
                if variable[var_samp]["cardinality"] == 2:
                    count[cstart[var_samp]] += v
                else:
                    count[cstart[var_samp] + v] += 1


@jit(nopython=True, cache=True, nogil=True)
def draw_sample(var_samp, var_copy, weight_copy, weight, variable, factor,
                fstart, fmap, vstart, vmap, equalPred, Z, var_value,
                weight_value):
    cardinality = variable[var_samp]["cardinality"]
    for value in range(cardinality):
        Z[value] = np.exp(potential(var_samp, value, var_copy, weight_copy,
                                    weight, variable, factor, fstart, fmap,
                                    vstart, vmap, equalPred, Z, var_value,
                                    weight_value))

    for j in range(1, cardinality):
        Z[j] += Z[j - 1]

    z = np.random.rand() * Z[cardinality - 1]

    # TODO: this looks at the full vector, slow if one var has high cardinality
    return np.argmax(Z >= z)


@jit(nopython=True, cache=True, nogil=True)
def potential(var_samp, value, var_copy, weight_copy, weight, variable, factor,
              fstart, fmap, vstart, vmap, equalPred, var_value, weight_value):
    p = 0.0
    for k in range(vstart[var_samp], vstart[var_samp + 1]):
        factor_id = vmap[k]
        p += weight_value[weight_copy][factor[vmap[k]]["weightId"]] * \
            eval_factor(factor_id, var_samp, value, var_copy, variable,
                        factor, fstart, fmap, equalPred, var_value)
    return p


@jit(nopython=True, cache=True, nogil=True)
def eval_factor(factor_id, var_samp, value, var_copy, variable, factor,
                fstart, fmap, equalPred, var_value):
    if factor[factor_id]["factorFunction"] == 0:  # FUNC_IMPLY_NATURAL
        for l in range(fstart[factor_id], fstart[factor_id + 1] - 1):
            v = value if (fmap[l] == var_samp) \
                else var_value[var_copy][fmap[l]]
            if v == 0:
                # Early return if body is not satisfied
                return 0

        # If this point is reached, body must be true
        head = value if (fmap[l] == var_samp) \
            else var_value[var_copy][fmap[factor_id + 1] - 1]
        if head:
            return 1
        return -1
    elif factor[factor_id]["factorFunction"] == 1:  # FUNC_OR
        for l in range(fstart[factor_id], fstart[factor_id + 1]):
            v = value if (fmap[l] == var_samp) \
                else var_value[var_copy][fmap[l]]
            if v == 1:
                return 1
        return -1
    elif factor[factor_id]["factorFunction"] == 3:  # FUNC_EQUAL
        v = value if (fmap[fstart[factor_id]] == var_samp) \
            else var_value[var_copy][fmap[fstart[factor_id]]]
        for l in range(fstart[factor_id] + 1, fstart[factor_id + 1]):
            w = value if (fmap[l] == var_samp) \
                else var_value[var_copy][fmap[l]]
            if v != w:
                return -1
        return 1
    elif factor[factor_id]["factorFunction"] == 2 \
            or factor[factor_id]["factorFunction"] == 4:
        # FUNC_AND or FUNC_ISTRUE
        for l in range(fstart[factor_id], fstart[factor_id + 1]):
            v = value if (fmap[l] == var_samp) \
                else var_value[var_copy][fmap[l]]
            if v == 0:
                return -1
        return 1
    elif factor[factor_id]["factorFunction"] == 7:  # FUNC_LINEAR
        res = 0
        head = value if (fmap[l] == var_samp) \
            else var_value[var_copy][fmap[factor_id + 1] - 1]
        for l in range(fstart[factor_id], fstart[factor_id + 1] - 1):
            v = value if (fmap[l] == var_samp) \
                else var_value[var_copy][fmap[l]]
            if v == head:
                res += 1
        # This does not match Dimmwitted, but matches the eq in the paper
        return res
    elif factor[factor_id]["factorFunction"] == 8:  # FUNC_RATIO
        res = 1
        head = value if (fmap[l] == var_samp) \
            else var_value[var_copy][fmap[factor_id + 1] - 1]
        for l in range(fstart[factor_id], fstart[factor_id + 1] - 1):
            v = value if (fmap[l] == var_samp) \
                else var_value[var_copy][fmap[l]]
            if v == head:
                res += 1
        # This does not match Dimmwitted, but matches the eq in the paper
        return math.log(res)  # TODO: use log2?
    elif factor[factor_id]["factorFunction"] == 9:  # FUNC_LOGICAL
        head = value if (fmap[l] == var_samp) \
            else var_value[var_copy][fmap[factor_id + 1] - 1]
        for l in range(fstart[factor_id], fstart[factor_id + 1] - 1):
            v = value if (fmap[l] == var_samp) \
                else var_value[var_copy][fmap[l]]
            if v == head:
                return 1
        return 0
    elif factor[factor_id]["factorFunction"] == 12:  # FUNC_AND_CATEGORICAL
        # TODO
        pass
    elif factor[factor_id]["factorFunction"] == 13:  # FUNC_IMPLY_MLN
        for l in range(fstart[factor_id], fstart[factor_id + 1] - 1):
            v = value if (fmap[l] == var_samp) \
                else var_value[var_copy][fmap[l]]
            if v == 0:
                # Early return if body is not satisfied
                return 1

        # If this point is reached, body must be true
        head = value if (fmap[l] == var_samp) \
            else var_value[var_copy][fmap[factor_id + 1] - 1]
        if head:
            return 1
        return 0
    else:  # FUNC_UNDEFINED
        print("Error: Factor Function", factor[factor_id]["factorFunction"],
              "( used in factor", factor_id, ") is not implemented.")
        raise NotImplementedError("Factor function is not implemented.")
