import numbskull
from numbskull.numbskulltypes import *
import numbskull.inference
import numpy as np
import codecs

# Commands from master to minions (Tags)
ASSIGN_ID = 'ASSIGN_ID'
INIT_NS = 'INIT_NS'
LOAD_FG = 'LOAD_FG'
SYNC_MAPPING = 'SYNC_MAPPING'
LEARN = 'LEARN'
INFER = 'INFER'

# Responses from minions to master (Tags)
ASSIGN_ID_RES = 'ASSIGN_ID_RES'
INIT_NS_RES = 'INIT_NS_RES'
LOAD_FG_RES = 'LOAD_FG_RES'
SYNC_MAPPING_RES = 'SYNC_MAPPING_RES'
LEARN_RES = 'LEARN_RES'
INFER_RES = 'INFER_RES'

# TODO These should be in some sort of util package
def get_views(cur):
    cur.execute("SELECT table_name FROM INFORMATION_SCHEMA.views WHERE table_name LIKE '%_sharding' AND table_schema = ANY (current_schemas(false))")
    view = []
    while True:
        temp = cur.fetchmany()
        if temp == []:
            break
        for i in temp:
            assert(len(i) == 1)
            view += i

    factor_view = []
    variable_view = []
    weight_view = []

    for v in view:
        is_f = ("_factors_" in v)
        is_v = ("_variables_" in v)
        is_w = ("_weights_" in v)
        assert((is_f + is_v + is_w) == 1)

        if is_f:
            factor_view.append(v)
        if is_v:
            variable_view.append(v)
        if is_w:
            weight_view.append(v)

    return (factor_view, variable_view, weight_view)

def read_factor_views(cur, views, sql_filter="True"):
    data = []
    op_template = "SELECT * FROM {table_name} " \
                  "WHERE {filter}"

    for v in views:
        # Find factor function
        ff = -1
        for (key, value) in numbskull.inference.FACTORS.items():
            if ("_" + key + "_").lower() in v:
                assert(ff == -1)
                ff = value
        # TODO: assume istrue if not found?
        #assert(ff != -1)
        if ff == -1:
            ff = numbskull.inference.FUNC_ISTRUE

        op = op_template.format(table_name=v, filter=sql_filter)
        cur.execute(op)
        while True:
            temp = cur.fetchmany()
            if temp == []:
                break
            for i in temp:
                data.append((i[:-3], i[-3], i[-2], i[-1], ff))
    return data

# views for variables and factors
def read_views(cur, views, sql_filter="True"):
    data = []
    op_template = "SELECT * FROM {table_name} " \
                  "WHERE {filter}"

    for v in views:
        op = op_template.format(table_name=v, filter=sql_filter)
        cur.execute(op)
        while True:
            temp = cur.fetchmany()
            if temp == []:
                break
            data += temp
    return data

def inverse_map(forward, index):
    # TODO: should probably also check that nothing is duplicated?
    ans = np.searchsorted(forward, index)
    assert(forward[ans] == index)
    return ans

def get_fg_data(cur, filt):
    (factor_view, variable_view, weight_view) = get_views(cur)
    
    # Load factors
    factor_data = read_factor_views(cur, factor_view, filt)
    
    factor = np.zeros(len(factor_data), Factor)
    factor_pt = np.zeros(len(factor_data), np.str_) # partition type
    factor_pid = np.zeros(len(factor_data), np.int64) # partition id
    
    for (i, f) in enumerate(factor_data):
        factor[i]["factorFunction"] = f[4]
        factor[i]["weightId"] = f[1]
        factor[i]["featureValue"] = f[2]
        factor[i]["arity"] = len(f[0])
        factor_pt[i] = f[3][0] # only first char needed now
                               # (partition id will match)
        factor_pid[i] = -1
        if f[3][1:] != "":
            factor_pid[i] = int(f[3][1:])
    
    if len(factor) > 0:
        factor[0]["ftv_offset"] = 0
    for i in range(1, len(factor)):
        factor[i]["ftv_offset"] = factor[i - 1]["ftv_offset"] + factor[i - 1]["arity"] 
    
    edges = 0
    if len(factor) > 0:
        edges = factor[-1]["ftv_offset"] + factor[-1]["arity"]
    fmap = np.zeros(edges, FactorToVar)
    
    index = 0
    for i in factor_data:
        for j in i[0]:
            fmap[index]["vid"] = j
            fmap[index]["dense_equal_to"] = 0 # TODO: how to actually get categorical info?
            index += 1
    
    # Load variable info
    var_data = read_views(cur, variable_view, filt)
    
    vid = np.zeros(len(var_data), np.int64)
    variable = np.zeros(len(var_data), Variable)
    var_pt = np.zeros(len(var_data), np.str_) # partition type
    var_pid = np.zeros(len(var_data), np.int64) # partition id
    for (i, v) in enumerate(var_data):
        vid[i] = v[0]
        variable[i]["isEvidence"] = v[1]
        variable[i]["initialValue"] = v[2]
        variable[i]["dataType"] = v[3]
        variable[i]["cardinality"] = v[4]
        #variable[i]["vtf_offset"] = ???
        var_pt[i] = v[5][0] # only first char needed now
                            # (partition id will match)
        var_pid[i] = -1
        if v[5][1:] != "":
            var_pid[i] = int(v[5][1:])
    
    perm = vid.argsort()
    vid = vid[perm]
    variable = variable[perm]
    var_pt = var_pt[perm]
    var_pid = var_pid[perm]
    
    # remap factor to variable
    for i in range(len(fmap)):
        fmap[i]["vid"] = inverse_map(vid, fmap[i]["vid"])
    
    # Load weight info
    # No filter since weights do not have a partition id
    weight_data = read_views(cur, weight_view, True)
    
    # TODO: if we start partitioning weights, these few lines
    # will have to be modified to be more like variables
    # (will need to remap weight_id's)
    # TODO: also need to remap weight id associated with
    # each factor
    weight = np.zeros(len(weight_data), Weight)
    for w in weight_data:
        wid = w[0]
        weight[wid]["isFixed"] = w[1]
        weight[wid]["initialValue"] = w[2]
    
    domain_mask = np.full(len(variable), True, np.bool)

    return (weight, variable, factor, fmap, domain_mask, edges, var_pt, var_pid, factor_pt, factor_pid, vid)

def serialize(array):
    return array.tobytes().decode('utf16').encode('utf8')

def deserialize(array, dtype):
    ar = array.decode('utf8').encode('utf16').lstrip(codecs.BOM_UTF16)
    return np.fromstring(ar, dtype=dtype)

