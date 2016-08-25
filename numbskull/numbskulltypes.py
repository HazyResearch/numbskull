import numpy as np
Meta = np.dtype([('weights',        np.int64),
                 ('variables',      np.int64),
                 ('factors',        np.int64),
                 ('edges',          np.int64)])

Weight = np.dtype([("isFixed",      np.bool),
                   ("initialValue", np.float64)])

Variable = np.dtype([("isEvidence",   np.int8),
                     ("initialValue", np.int32),
                     ("dataType",     np.int16),
                     ("cardinality",  np.int32)])

Factor = np.dtype([("factorFunction", np.int16),
                   ("weightId",       np.int64),
                   ("featureValue",   np.float64)])
