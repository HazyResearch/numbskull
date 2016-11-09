"""TODO."""

from __future__ import print_function, absolute_import
import numba
from numba import jit
import numpy as np


# HELPER METHODS #
def dataType(i):
    """TODO."""
    return {0: "Boolean",
            1: "Categorical"}.get(i, "Unknown")


@jit(nopython=True, cache=True)
def compute_var_map(variables, factors, fmap, vmap, factor_index, domain_mask,
                    factors_to_skip=np.empty(0, np.int64)):
    """TODO."""
    # Fill in domain values (for mapping when dumping marginals)
    for i, v in enumerate(variables):
        # skip boolean (value is 0)
        if v["dataType"] == 0:
            continue  # default to 0
        # categorical with explicit domain
        if domain_mask[i]:
            continue  # filled already
        else:  # categorical with implicit domain [0...cardinality)
            for index in range(v["cardinality"]):
                vmap[v["vtf_offset"] + index]["value"] = index

    # Fill in factor_index and indexes into factor_index
    # Step 1: populate VTF.length
    for ftv in fmap:
        vid = ftv["vid"]
        val = ftv["dense_equal_to"] if variables[vid]["dataType"] == 1 else 0
        vtf = vmap[variables[vid]["vtf_offset"] + val]
        vtf["factor_index_length"] += 1

    # Step 2: populate VTF.offset
    last_len = 0
    last_off = 0
    for i, vtf in enumerate(vmap):
        vtf["factor_index_offset"] = last_off + last_len
        last_len = vtf["factor_index_length"]
        last_off = vtf["factor_index_offset"]

    # Step 3: populate factor_index
    offsets = vmap["factor_index_offset"].copy()
    fts_index = 0  # factors_to_skip index
    for i, fac in enumerate(factors):
        if fts_index < len(factors_to_skip) and \
           factors_to_skip[fts_index] == i:
            fts_index += 1
            continue

        for j in range(fac["ftv_offset"], fac["ftv_offset"] + fac["arity"]):
            ftv = fmap[j]
            vid = ftv["vid"]
            val = ftv["dense_equal_to"] if variables[
                vid]["dataType"] == 1 else 0
            vtf_idx = variables[vid]["vtf_offset"] + val
            fidx = offsets[vtf_idx]
            factor_index[fidx] = i
            offsets[vtf_idx] += 1

    # Step 4: remove dupes from factor_index
    for vtf in vmap:
        offset = vtf["factor_index_offset"]
        length = vtf["factor_index_length"]
        new_list = factor_index[offset: offset + length]
        new_list.sort()
        i = 0
        last_fid = -1
        for fid in new_list:
            if last_fid == fid:
                continue
            last_fid = fid
            factor_index[offset + i] = fid
            i += 1
        vtf["factor_index_length"] = i


@jit(nopython=True, cache=True)
def reverse(data, start, end):
    """TODO."""
    end -= 1
    while (start < end):
        data[start], data[end] = data[end], data[start]
        start += 1
        end -= 1


@jit(nopython=True, cache=True)
def reverse_array(data):
    """TODO."""
    # TODO: why does this fail?
    # data = np.flipud(data)
    reverse(data, 0, data.size)


# DEFINE NUMBA-BASED DATA LOADING METHODS #
@jit(nopython=True, cache=True)
def load_weights(data, nweights, weights):
    """TODO."""
    for i in range(nweights):
        # TODO: read types from struct?
        # TODO: byteswap only if system is little-endian

        buf = data[(17 * i):(17 * i + 8)]
        reverse_array(buf)
        weightId = np.frombuffer(buf, dtype=np.int64)[0]

        isFixed = data[17 * i + 8]

        buf = data[(17 * i + 9):(17 * i + 17)]
        reverse_array(buf)
        initialValue = np.frombuffer(buf, dtype=np.float64)[0]

        weights[weightId]["isFixed"] = isFixed
        weights[weightId]["initialValue"] = initialValue

    print("LOADED WEIGHTS")


@jit(nopython=True, cache=True)
def load_variables(data, nvariables, variables):
    """TODO."""
    for i in range(nvariables):
        # TODO: read types from struct?
        # TODO: byteswap only if system is little-endian

        buf = data[(27 * i):(27 * i + 8)]
        reverse_array(buf)
        variableId = np.frombuffer(buf, dtype=np.int64)[0]

        isEvidence = data[27 * i + 8]

        buf = data[(27 * i + 9):(27 * i + 17)]
        reverse_array(buf)
        initialValue = np.frombuffer(buf, dtype=np.int64)[0]

        buf = data[(27 * i + 17):(27 * i + 19)]
        reverse_array(buf)
        dataType = np.frombuffer(buf, dtype=np.int16)[0]

        buf = data[(27 * i + 19):(27 * i + 27)]
        reverse_array(buf)
        cardinality = np.frombuffer(buf, dtype=np.int64)[0]

        variables[variableId]["isEvidence"] = isEvidence
        variables[variableId]["initialValue"] = initialValue
        variables[variableId]["dataType"] = dataType
        variables[variableId]["cardinality"] = cardinality

    print("LOADED VARS")


@jit(nopython=True, cache=True)
def load_domains(data, domain_mask, vmap, variables):
    """TODO."""
    index = 0
    while index < data.size:
        buf = data[index: index + 8]
        reverse_array(buf)
        variableId = np.frombuffer(buf, dtype=np.int64)[0]
        index += 8

        buf = data[index: index + 8]
        reverse_array(buf)
        cardinality = np.frombuffer(buf, dtype=np.int64)[0]
        index += 8

        domain_mask[variableId] = True

        # NOTE: values are sorted already by DD
        for j in range(cardinality):
            buf = data[index: index + 8]
            reverse_array(buf)
            val = np.frombuffer(buf, dtype=np.int64)[0]
            index += 8
            vmap[variables[variableId]["vtf_offset"] + j]["value"] = val
            # translate initial value into dense index
            if val == variables[variableId]["initialValue"]:
                variables[variableId]["initialValue"] = j

    print("LOADED DOMAINS")


@jit(nopython=True, cache=True)
def load_factors(data, nfactors, factors, fmap, domain_mask, variable, vmap):
    """TODO."""
    index = 0
    fmap_idx = 0
    k = 0  # somehow numba 0.28 would raise LowerError without this line
    for i in range(nfactors):
        buf = data[index:(index + 2)]
        reverse_array(buf)
        factors[i]["factorFunction"] = np.frombuffer(buf, dtype=np.int16)[0]

        buf = data[(index + 2):(index + 10)]
        reverse_array(buf)
        arity = np.frombuffer(buf, dtype=np.int64)[0]
        factors[i]["arity"] = arity
        factors[i]["ftv_offset"] = fmap_idx

        index += 10  # TODO: update index once per loop?

        for k in range(arity):
            buf = data[index:(index + 8)]
            reverse_array(buf)
            vid = np.frombuffer(buf, dtype=np.int64)[0]
            fmap[fmap_idx + k]["vid"] = vid

            buf = data[(index + 8):(index + 16)]
            reverse_array(buf)
            val = np.frombuffer(buf, dtype=np.int64)[0]
            # translate initial value into dense index using bisect
            if domain_mask[vid]:
                start = variable[vid]["vtf_offset"]
                end = start + variable[vid]["cardinality"]
                val = np.searchsorted(vmap["value"][start:end], val)
            fmap[fmap_idx + k]["dense_equal_to"] = val
            index += 16
        fmap_idx += arity

        buf = data[index:(index + 8)]
        reverse_array(buf)
        factors[i]["weightId"] = np.frombuffer(buf, dtype=np.int64)[0]

        buf = data[(index + 8):(index + 16)]
        reverse_array(buf)
        factors[i]["featureValue"] = np.frombuffer(buf, dtype=np.float64)[0]

        index += 16

    print("LOADED FACTORS")
