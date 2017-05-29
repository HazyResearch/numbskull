"""TODO."""

from __future__ import print_function, absolute_import
import numba
from numba import jit
import numpy as np
import math


@jit(nopython=True, cache=True, nogil=True)
def gibbsthread(shardID, nshards, var_copy, weight_copy, weight, variable,
                factor, fmap, vmap, factor_index, Z, cstart,
                count, var_value, weight_value, sample_evidence, burnin):
    """TODO."""
    # Indentify start and end variable
    nvar = variable.shape[0]
    start = (shardID * nvar) // nshards
    end = ((shardID + 1) * nvar) // nshards
    # TODO: give option do not store result, or just store tally
    for var_samp in range(start, end):
        if variable[var_samp]["isEvidence"] == 4:
            # This variable is not owned by this machine
            continue
        if variable[var_samp]["isEvidence"] == 0 or sample_evidence:
            v = draw_sample(var_samp, var_copy, weight_copy, weight, variable,
                            factor, fmap, vmap, factor_index, Z[shardID],
                            var_value, weight_value)
            var_value[var_copy][var_samp] = v
            if not burnin:
                if variable[var_samp]["cardinality"] == 2:
                    count[cstart[var_samp]] += v
                else:
                    count[cstart[var_samp] + v] += 1


@jit(nopython=True, cache=True, nogil=True)
def draw_sample(var_samp, var_copy, weight_copy, weight, variable, factor,
                fmap, vmap, factor_index, Z, var_value, weight_value):
    """TODO."""
    cardinality = variable[var_samp]["cardinality"]
    for value in range(cardinality):
        Z[value] = np.exp(potential(var_samp, value, var_copy, weight_copy,
                                    weight, variable, factor, fmap,
                                    vmap, factor_index, var_value,
                                    weight_value))

    for j in range(1, cardinality):
        Z[j] += Z[j - 1]

    z = np.random.rand() * Z[cardinality - 1]

    return np.argmax(Z[:cardinality] >= z)


@jit(nopython=True, cache=True, nogil=True)
def potential(var_samp, value, var_copy, weight_copy, weight, variable, factor,
              fmap, vmap, factor_index, var_value, weight_value):
    """TODO."""
    p = 0.0
    varval_off = value
    if variable[var_samp]["dataType"] == 0:
        varval_off = 0
    vtf = vmap[variable[var_samp]["vtf_offset"] + varval_off]
    start = vtf["factor_index_offset"]
    end = start + vtf["factor_index_length"]
    for k in range(start, end):
        factor_id = factor_index[k]
        p += weight_value[weight_copy][factor[factor_id]["weightId"]] * \
            eval_factor(factor_id, var_samp, value, var_copy, variable,
                        factor, fmap, var_value)
    return p


FACTORS = {
    # Factor functions for boolean variables
    "NOOP": -1,
    "IMPLY_NATURAL": 0,
    "OR": 1,
    "EQUAL": 3,
    "AND": 2,
    "ISTRUE": 4,
    "LINEAR": 7,
    "RATIO": 8,
    "LOGICAL": 9,
    "IMPLY_MLN": 13,

    # Factor functions for categorical variables
    "AND_CAT": 12,
    "OR_CAT": 14,
    "EQUAL_CAT_CONST": 15,
    "IMPLY_NATURAL_CAT": 16,
    "IMPLY_MLN_CAT": 17,

    # Factor functions for generative models for data programming.
    #
    # These functions accept two types of categorical variables:
    #
    # y \in {1, -1} corresponding to latent labels, and
    # l \in {1, 0, -1} corresponding to labeling function outputs.
    #
    # The values of y are mapped to Numbskull variables y_index
    #     via {-1: 0, 1: 1}, and
    # the values of l are mapped to Numbskull variables l_index
    #     via {-1: 0, 0: 1, 1: 2}.

    # h(y) := y
    "DP_GEN_CLASS_PRIOR": 18,

    # h(l) := l
    "DP_GEN_LF_PRIOR": 19,

    # h(l) := l * l
    "DP_GEN_LF_PROPENSITY": 20,

    # h(y, l) := y * l
    "DP_GEN_LF_ACCURACY": 21,

    # h(l) := y * l * l
    "DP_GEN_LF_CLASS_PROPENSITY": 22,

    # l_2 fixes errors made by l_1
    #
    # h(y, l_1, l_2) := if l_1 == 0 and l_2 != 0: -1,
    #                   elif l_1 == -1 * y and l_2 == y: 1,
    #                   else: 0
    "DP_GEN_DEP_FIXING": 23,

    # l_2 reinforces the output of l_1
    #
    # h(y, l_1, l_2) := if l_1 == 0 and l_2 != 0: -1,
    #                   elif l_1 == y and l_2 == y: 1,
    #                   else: 0
    "DP_GEN_DEP_REINFORCING": 24,

    # h(l_1, l_2) := if l_1 != 0 and l_2 != 0: -1, else: 0
    "DP_GEN_DEP_EXCLUSIVE": 25,

    #h(l_1, l_2) := if l_1 == l_2: 1, else: 0
    "DP_GEN_DEP_SIMILAR": 26,

    # Factor functions for distribution
    "UFO": 30
}

for (key, value) in FACTORS.items():
    exec("FUNC_" + key + " = " + str(value))


@jit(nopython=True, cache=True, nogil=True)
def eval_factor(factor_id, var_samp, value, var_copy, variable, factor, fmap,
                var_value):
    """TODO."""
    ####################
    # BINARY VARIABLES #
    ####################
    fac = factor[factor_id]
    ftv_start = fac["ftv_offset"]
    ftv_end = ftv_start + fac["arity"]

    if fac["factorFunction"] == FUNC_NOOP:
        return 0
    elif fac["factorFunction"] == FUNC_IMPLY_NATURAL:
        for l in range(ftv_start, ftv_end):
            v = value if (fmap[l]["vid"] == var_samp) else \
                var_value[var_copy][fmap[l]["vid"]]
            if v == 0:
                # Early return if body is not satisfied
                return 0

        # If this point is reached, body must be true
        l = ftv_end - 1
        head = value if (fmap[l]["vid"] == var_samp) else \
            var_value[var_copy][fmap[l]["vid"]]
        if head:
            return 1
        return -1
    elif factor[factor_id]["factorFunction"] == FUNC_OR:
        for l in range(ftv_start, ftv_end):
            v = value if (fmap[l]["vid"] == var_samp) else \
                var_value[var_copy][fmap[l]["vid"]]
            if v == 1:
                return 1
        return -1
    elif factor[factor_id]["factorFunction"] == FUNC_EQUAL:
        v = value if (fmap[ftv_start]["vid"] == var_samp) \
            else var_value[var_copy][fmap[ftv_start]["vid"]]
        for l in range(ftv_start + 1, ftv_end):
            w = value if (fmap[l]["vid"] == var_samp) \
                else var_value[var_copy][fmap[l]["vid"]]
            if v != w:
                return -1
        return 1
    elif factor[factor_id]["factorFunction"] == FUNC_AND \
            or factor[factor_id]["factorFunction"] == FUNC_ISTRUE:
        for l in range(ftv_start, ftv_end):
            v = value if (fmap[l]["vid"] == var_samp) \
                else var_value[var_copy][fmap[l]["vid"]]
            if v == 0:
                return -1
        return 1
    elif factor[factor_id]["factorFunction"] == FUNC_LINEAR:
        res = 0
        head = value if (fmap[ftv_end - 1]["vid"] == var_samp) \
            else var_value[var_copy][fmap[ftv_end - 1]["vid"]]
        for l in range(ftv_start, ftv_end - 1):
            v = value if (fmap[l]["vid"] == var_samp) \
                else var_value[var_copy][fmap[l]["vid"]]
            if v == head:
                res += 1
        # This does not match Dimmwitted, but matches the eq in the paper
        return res
    elif factor[factor_id]["factorFunction"] == FUNC_RATIO:
        res = 1
        head = value if (fmap[ftv_end - 1]["vid"] == var_samp) \
            else var_value[var_copy][fmap[ftv_end - 1]["vid"]]
        for l in range(ftv_start, ftv_end - 1):
            v = value if (fmap[l]["vid"] == var_samp) \
                else var_value[var_copy][fmap[l]["vid"]]
            if v == head:
                res += 1
        # This does not match Dimmwitted, but matches the eq in the paper
        return math.log(res)  # TODO: use log2?
    elif factor[factor_id]["factorFunction"] == FUNC_LOGICAL:
        head = value if (fmap[ftv_end - 1]["vid"] == var_samp) \
            else var_value[var_copy][fmap[ftv_end - 1]["vid"]]
        for l in range(ftv_start, ftv_end - 1):
            v = value if (fmap[l]["vid"] == var_samp) \
                else var_value[var_copy][fmap[l]["vid"]]
            if v == head:
                return 1
        return 0
    elif factor[factor_id]["factorFunction"] == FUNC_IMPLY_MLN:
        for l in range(ftv_start, ftv_end - 1):
            v = value if (fmap[l]["vid"] == var_samp) \
                else var_value[var_copy][fmap[l]["vid"]]
            if v == 0:
                # Early return if body is not satisfied
                return 1

        # If this point is reached, body must be true
        l = ftv_end - 1
        head = value if (fmap[l]["vid"] == var_samp) \
            else var_value[var_copy][l]
        if head:
            return 1
        return 0

    #########################
    # CATEGORICAL VARIABLES #
    #########################
    elif factor[factor_id]["factorFunction"] == FUNC_AND_CAT \
            or factor[factor_id]["factorFunction"] == FUNC_EQUAL_CAT_CONST:
        for l in range(ftv_start, ftv_end):
            v = value if (fmap[l]["vid"] == var_samp) \
                else var_value[var_copy][fmap[l]["vid"]]
            if v != fmap[l]["dense_equal_to"]:
                return 0
        return 1
    elif factor[factor_id]["factorFunction"] == FUNC_OR_CAT:
        for l in range(ftv_start, ftv_end):
            v = value if (fmap[l]["vid"] == var_samp) \
                else var_value[var_copy][fmap[l]["vid"]]
            if v == fmap[l]["dense_equal_to"]:
                return 1
        return -1
    elif factor[factor_id]["factorFunction"] == FUNC_IMPLY_NATURAL_CAT:
        for l in range(ftv_start, ftv_end - 1):
            v = value if (fmap[l]["vid"] == var_samp) \
                else var_value[var_copy][fmap[l]["vid"]]
            if v != fmap[l]["dense_equal_to"]:
                # Early return if body is not satisfied
                return 0

        # If this point is reached, body must be true
        l = ftv_end - 1
        head = value if (fmap[l]["vid"] == var_samp) \
            else var_value[var_copy][l]
        if head == fmap[l]["dense_equal_to"]:
            return 1
        return -1
    elif factor[factor_id]["factorFunction"] == FUNC_IMPLY_MLN_CAT:
        for l in range(ftv_start, ftv_end - 1):
            v = value if (fmap[l]["vid"] == var_samp) \
                else var_value[var_copy][fmap[l]["vid"]]
            if v != fmap[l]["dense_equal_to"]:
                # Early return if body is not satisfied
                return 1

        # If this point is reached, body must be true
        l = ftv_end - 1
        head = value if (fmap[l]["vid"] == var_samp) \
            else var_value[var_copy][l]
        if head == fmap[l]["dense_equal_to"]:
            return 1
        return 0

    #####################
    # DATA PROGRAMMING  #
    # GENERATIVE MODELS #
    #####################
    elif factor[factor_id]["factorFunction"] == FUNC_DP_GEN_CLASS_PRIOR:
        # NB: this doesn't make sense for categoricals
        y_index = value if fmap[ftv_start]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start]["vid"]]
        return 1 if y_index == 1 else -1
    elif factor[factor_id]["factorFunction"] == FUNC_DP_GEN_LF_PRIOR:
        # NB: this doesn't make sense for categoricals
        l_index = value if fmap[ftv_start]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start]["vid"]]
        if l_index == 2:
            return -1
        elif l_index == 0:
            return 0
        else:
            return 1
    elif factor[factor_id]["factorFunction"] == FUNC_DP_GEN_LF_PROPENSITY:
        l_index = value if fmap[ftv_start]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start]["vid"]]
        abstain = variable[fmap[ftv_start]["vid"]]["cardinality"] - 1
        return 0 if l_index == abstain else 1
    elif factor[factor_id]["factorFunction"] == FUNC_DP_GEN_LF_ACCURACY:
        y_index = value if fmap[ftv_start]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start]["vid"]]
        l_index = value if fmap[ftv_start + 1]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start + 1]["vid"]]
        abstain = variable[fmap[ftv_start + 1]["vid"]]["cardinality"] - 1
        if l_index == abstain:
            return 0
        elif y_index == l_index:
            return 1
        else:
            return -1
    elif factor[factor_id]["factorFunction"] == \
            FUNC_DP_GEN_LF_CLASS_PROPENSITY:
        # NB: this doesn't make sense for categoricals
        y_index = value if fmap[ftv_start]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start]["vid"]]
        l_index = value if fmap[ftv_start + 1]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start + 1]["vid"]]
        abstain = variable[fmap[ftv_start + 1]["vid"]]["cardinality"] - 1
        if l_index == abstain:
            return 0
        elif y_index == 1:
            return 1
        else:
            return -1
    elif factor[factor_id]["factorFunction"] == FUNC_DP_GEN_DEP_FIXING:
        # NB: this doesn't make sense for categoricals
        y_index = value if fmap[ftv_start]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start]["vid"]]
        l1_index = value if fmap[ftv_start + 1]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start + 1]["vid"]]
        l2_index = value if fmap[ftv_start + 2]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start + 2]["vid"]]
        abstain = variable[fmap[ftv_start + 1]["vid"]]["cardinality"] - 1
        if l1_index == abstain:
            return -1 if l2_index != 1 else 0
        elif l1_index == 0 and l2_index == 1 and y_index == 1:
            return 1
        elif l1_index == 1 and l2_index == 0 and y_index == 0:
            return 1
        else:
            return 0
    elif factor[factor_id]["factorFunction"] == FUNC_DP_GEN_DEP_REINFORCING:
        # NB: this doesn't make sense for categoricals
        y_index = value if fmap[ftv_start]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start]["vid"]]
        l1_index = value if fmap[ftv_start + 1]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start + 1]["vid"]]
        l2_index = value if fmap[ftv_start + 2]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start + 2]["vid"]]
        abstain = variable[fmap[ftv_start + 1]["vid"]]["cardinality"] - 1
        if l1_index == abstain:
            return -1 if l2_index != 1 else 0
        elif l1_index == 0 and l2_index == 0 and y_index == 0:
            return 1
        elif l1_index == 1 and l2_index == 1 and y_index == 1:
            return 1
        else:
            return 0
    elif factor[factor_id]["factorFunction"] == FUNC_DP_GEN_DEP_EXCLUSIVE:
        l1_index = value if fmap[ftv_start]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start]["vid"]]
        l2_index = value if fmap[ftv_start + 1]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start + 1]["vid"]]
        abstain = variable[fmap[ftv_start]["vid"]]["cardinality"] - 1
        return 0 if l1_index == abstain or l2_index == abstain else -1
    elif factor[factor_id]["factorFunction"] == FUNC_DP_GEN_DEP_SIMILAR:
        l1_index = value if fmap[ftv_start]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start]["vid"]]
        l2_index = value if fmap[ftv_start + 1]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start + 1]["vid"]]
        return 1 if l1_index == l2_index else 0

    ###########################################
    # FACTORS FOR OPTIMIZING DISTRIBUTED CODE #
    ###########################################
    elif factor[factor_id]["factorFunction"] == FUNC_UFO:
        v = value if fmap[ftv_start]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start]["vid"]]
        if v == 0:
            return 0

        return value if fmap[ftv_start + v - 1]["vid"] == var_samp else \
            var_value[var_copy][fmap[ftv_start + v - 1]["vid"]]

    ######################
    # FACTOR NOT DEFINED #
    ######################
    else:  # FUNC_UNDEFINED
        print("Error: Factor Function", factor[factor_id]["factorFunction"],
              "( used in factor", factor_id, ") is not implemented.")
        raise NotImplementedError("Factor function is not implemented.")
