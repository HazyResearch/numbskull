"""TODO."""

from __future__ import print_function, absolute_import
import numba
from numba import jit
import numpy as np
import math


# Search "USER" to find sections that need to be implemented.


# USER: insert name of UDF and cardinality here
UDF_CARDINALITY = {
    # UDFs for toy example
    "TOY_AND":    2,
    "TOY_DIRECT": 1,
    "TOY_DIRECT_ABSTAIN": 1,
    "TOY_DIRECT_SINGLE": 1
}

# Automatically select a unique index for each UDF (no modification needed)
UDF_INDEX = {}
UDF_NAME = {}
udf_index = 0
for udf in UDF_CARDINALITY:
    UDF_INDEX[udf] = udf_index
    UDF_NAME[udf_index] = udf
    udf_index += 1

# Create a constant for the index of each UDF
for (key, value) in UDF_INDEX.items():
    exec(key + " = " + str(value))

# USER: Specify the list of UDFs that are used in a single model.
UDF_USAGE = {
    "TOY": [TOY_AND, TOY_DIRECT, TOY_DIRECT],
    "DDSM": [TOY_DIRECT_ABSTAIN, TOY_DIRECT_ABSTAIN, TOY_DIRECT_ABSTAIN, TOY_DIRECT_ABSTAIN, TOY_DIRECT_SINGLE, TOY_DIRECT_ABSTAIN]
}

# USER: There are not modifications necessary here. However, the value
# generated in UDF_OFFSET needs to be given to CoralModel as L_offset.
UDF_OFFSET = {}
UDF_OFFSET_END = {}
UdfStart = np.empty(len(UDF_USAGE) + 1, np.int64) # UdfStart[i] = first index corresponding to application [i]
LfCount = np.empty(len(UDF_USAGE), np.int64) # LfCount[i] = number of LFs in application [i]
UdfCardinalityStart = np.empty(len(UDF_USAGE) + 1, np.int64)
UdfMap = np.empty(sum(len(value) for (key, value) in UDF_USAGE.items()), np.int64)
UdfCardinality = np.empty(sum(len(value) for (key, value) in UDF_USAGE.items()), np.int64)
index = 0
ci = 0
udf_offset = 1000
for (key, value) in UDF_USAGE.items():
    UDF_OFFSET[key] = udf_offset
    UdfStart[index] = udf_offset
    LfCount[index] = len(UDF_USAGE[key])
    UdfCardinalityStart[index] = ci
    for i in range(LfCount[index]):
        UdfMap[ci] = UDF_USAGE[key][i]
        UdfCardinality[ci] = UDF_CARDINALITY[UDF_NAME[UDF_USAGE[key][i]]]
        ci += 1
    udf_offset += len(UDF_USAGE[key]) # LF accuracy
    udf_offset += len(UDF_USAGE[key]) * (len(UDF_USAGE[key]) - 1) / 2 # correlations
    index += 1
    UdfStart[index] = udf_offset
    UDF_OFFSET_END[key] = udf_offset
    UdfCardinalityStart[index] = ci
    exec(key + "_UDF_OFFSET = " + str(UDF_OFFSET[key]))

# USER: Implement the UDF here
# The following code can be used to obtain the correct value of a variable:
# vi = value               if (fmap[ftv_start + i]["vid"] == var_samp) \
#     else var_value[var_copy][fmap[ftv_start + i]["vid"]]
@jit(nopython=True, cache=True, nogil=True)
def udf(udf_index, var_samp, value, var_copy, var_value, fmap, ftv_start):
    if udf_index == TOY_AND:
        v1 = value               if (fmap[ftv_start + 0]["vid"] == var_samp) \
            else var_value[var_copy][fmap[ftv_start + 0]["vid"]]
        v2 = value               if (fmap[ftv_start + 1]["vid"] == var_samp) \
            else var_value[var_copy][fmap[ftv_start + 1]["vid"]]
        if v1 == 1 and v2 == 1:
            return 1
        return -1
    elif udf_index == TOY_DIRECT:
        v1 = value               if (fmap[ftv_start + 0]["vid"] == var_samp) \
            else var_value[var_copy][fmap[ftv_start + 0]["vid"]]
        return 2 * v1 - 1
    elif udf_index == TOY_DIRECT_ABSTAIN:
        v1 = value               if (fmap[ftv_start + 0]["vid"] == var_samp) \
            else var_value[var_copy][fmap[ftv_start + 0]["vid"]]
        return v1-1
    elif udf_index == TOY_DIRECT_SINGLE:
        #print("Working")
        v1 = value               if (fmap[ftv_start + 0]["vid"] == var_samp) \
            else var_value[var_copy][fmap[ftv_start + 0]["vid"]]
        return v1-1
    else:
        print("Error: UDF", udf_index,
              "is not implemented.")
        raise NotImplementedError("UDF is not implemented.")