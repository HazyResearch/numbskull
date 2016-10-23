"""TODO."""

from __future__ import print_function, absolute_import
import numpy as np

# TODO (shared with DW): space optimization:
# 1. use smaller ints for some fields
# 2. replace a[x].length with a[x+1].offset - a[x].offset


Meta = np.dtype([('weights', np.int64),
                 ('variables', np.int64),
                 ('factors', np.int64),
                 ('edges', np.int64)])

Weight = np.dtype([("isFixed", np.bool),
                   ("initialValue", np.float64)])

Variable = np.dtype([("isEvidence", np.int8),
                     ("initialValue", np.int64),
                     ("dataType", np.int16),
                     ("cardinality", np.int64),
                     ("vtf_offset", np.int64)])

Factor = np.dtype([("factorFunction", np.int16),
                   ("weightId", np.int64),
                   ("featureValue", np.float64),
                   ("arity", np.int64),
                   ("ftv_offset", np.int64)])

FactorToVar = np.dtype([("vid", np.int64),
                        ("dense_equal_to", np.int64)])

VarToFactor = np.dtype([("value", np.int64),
                        ("factor_index_offset", np.int64),
                        ("factor_index_length", np.int64)])

UnaryFactorOpt = np.dtype([('vid', np.int64),
                           ('weightId', np.int64)])
