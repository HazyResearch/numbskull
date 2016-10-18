"""TODO."""

from __future__ import print_function
import numbskull
from numbskull.numbskulltypes import *
import numbskull.inference
import numpy as np
import codecs
import numba
import time

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
    """TODO."""
    cur.execute("SELECT table_name "
                "FROM INFORMATION_SCHEMA.views "
                "WHERE table_name LIKE '%_sharding' "
                "  AND table_schema = ANY (current_schemas(false))")
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


@numba.jit(nopython=True, cache=True, nogil=True)
def get_factors_helper(row, ff, factor, factor_pt, fmap, factor_index,
                       fmap_index):
    """TODO."""
    for i in row:
        factor[factor_index]["factorFunction"] = ff
        factor[factor_index]["weightId"] = i[-3]
        factor[factor_index]["featureValue"] = i[-2]
        factor_pt[factor_index] = i[-1]
        factor[factor_index]["arity"] = len(i) - 3
        if factor_index == 0:
            factor[factor_index]["ftv_offset"] = 0
        else:
            factor[factor_index]["ftv_offset"] = \
                factor[factor_index - 1]["ftv_offset"] + \
                factor[factor_index - 1]["arity"]
        factor_index += 1

        for j in i[:-3]:
            fmap[fmap_index]["vid"] = j
            # TODO: how to actually get categorical info?
            fmap[fmap_index]["dense_equal_to"] = 0
            fmap_index += 1

    return factor_index, fmap_index


def get_factors(cur, views, sql_filter="True"):
    """TODO."""
    factors = 0
    edges = 0

    # This operation is for counting rows, and getting rows
    op_template = "SELECT {cmd} FROM {table_name} WHERE {filter}"

    # This operation is for counting columns in a table
    count_template = "SELECT COUNT(*) " \
                     "FROM INFORMATION_SCHEMA.COLUMNS " \
                     "WHERE table_schema = 'public' " \
                     "AND table_name = '{table_name}'"

    # This operation is for getting columns in a table
    names_template = "SELECT column_name " \
                     "FROM INFORMATION_SCHEMA.COLUMNS " \
                     "WHERE table_schema = 'public' " \
                     "AND table_name = '{table_name}' " \
                     "ORDER BY ordinal_position"

    # Pre-count number of factors and edges
    # TODO: can this step be avoided?
    for table in views:
        op = op_template.format(cmd="COUNT(*)", table_name=table,
                                filter=sql_filter)
        cur.execute(op)
        f = cur.fetchone()[0]  # number of factors in this table

        count = count_template.format(table_name=table)
        cur.execute(count)
        v = cur.fetchone()[0] - 3  # number of vars used by these factors

        factors += f
        edges += f * v

    factor = np.zeros(factors, Factor)
    factor_pt = np.zeros(factors, np.int8)  # partition type
    fmap = np.zeros(edges, FactorToVar)

    factor_index = 0
    fmap_index = 0
    for v in views:
        # Find factor function
        ff = -1
        for (key, value) in numbskull.inference.FACTORS.items():
            if ("_" + key + "_").lower() in v:
                assert(ff == -1)
                ff = value
        # TODO: assume istrue if not found?
        if ff == -1:
            ff = numbskull.inference.FUNC_ISTRUE

        names_op = names_template.format(table_name=v)
        cur.execute(names_op)
        name = cur.fetchall()
        for i in range(len(name)):
            assert(len(name[i]) == 1)
            name[i] = name[i][0]
        assert(name[-3] == "weight_id")
        assert(name[-2] == "feature_value")
        assert(name[-1] == "partition_key")
        cmd = ", ".join(['"' + i + '"' for i in name[:-1]]) + ", ASCII(LEFT(partition_key, 1))"

        op = op_template.format(cmd=cmd, table_name=v, filter=sql_filter)
        cur.execute(op)
        while True:
            row = cur.fetchmany(10000)
            if row == []:
                break
            (factor_index, fmap_index) = \
                get_factors_helper(row, ff, factor, factor_pt, fmap,
                                   factor_index, fmap_index)

    return factor, factor_pt.view('c'), fmap


@numba.jit(nopython=True, cache=True, nogil=True)
def get_variables_helper(row, vid, variable, var_pt, var_pid, index):
    for v in row:
        vid[index] = v[0]
        variable[index]["isEvidence"] = v[1]
        variable[index]["initialValue"] = v[2]
        variable[index]["dataType"] = v[3]
        variable[index]["cardinality"] = v[4]
        var_pt[index] = v[5]
        index += 1
    return index


def get_variables(cur, views, sql_filter="True"):
    """TODO."""
    op_template = "SELECT {cmd} FROM {table_name} " \
                  "WHERE {filter}"

    # Obtain count of variables
    # TODO: is there a way to do this together with next part?
    #       (one query per table)
    n = 0
    for v in views:
        op = op_template.format(cmd="COUNT(*)", table_name=v, filter=sql_filter)
        op = op_template.format(cmd="COUNT(*)", table_name=v, filter=sql_filter)
        cur.execute(op)
        n += cur.fetchone()[0]  # number of factors in this table

    vid = np.zeros(n, np.int64)
    variable = np.zeros(n, Variable)
    var_pt = np.zeros(n, np.int8)  # partition type
    var_pid = np.zeros(n, np.int64)  # partition id

    index = 0
    for v in views:
        cmd = "vid, variable_role, init_value, variable_type, cardinality, " \
              "ASCII(LEFT(partition_key, 1))"
        op = op_template.format(cmd=cmd, table_name=v, filter=sql_filter)
        cur.execute(op)
        while True:
            row = cur.fetchmany(10000)
            if row == []:
                break
            index = get_variables_helper(row, vid, variable, var_pt, var_pid, index)

    perm = vid.argsort()
    vid = vid[perm]
    variable = variable[perm]
    var_pt = var_pt[perm]
    var_pid = var_pid[perm]

    return vid, variable, var_pt.view('c'), var_pid


def read_factor_views(cur, views, sql_filter="True"):
    """TODO."""
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
        # assert(ff != -1)
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
    """TODO."""
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


@numba.jit(nopython=True, cache=True, nogil=True)
def inverse_map(forward, index):
    """TODO."""
    # TODO: should probably also check that nothing is duplicated?
    ans = np.searchsorted(forward, index)
    assert(forward[ans] == index)
    return ans


@numba.jit(nopython=True, cache=True, nogil=True)
def remap_fmap(fmap, vid):
    for i in range(len(fmap)):
        fmap[i]["vid"] = inverse_map(vid, fmap[i]["vid"])


def get_fg_data(cur, filt):
    """TODO."""
    print("***GET_FG_DATA***")
    time1 = time.time()
    (factor_view, variable_view, weight_view) = get_views(cur)
    time2 = time.time()
    print("get_views: " + str(time2 - time1))

    (factor2, factor_pt2, fmap2) = get_factors(cur, factor_view, filt)
    time1 = time2
    time2 = time.time()
    print("get_factors: " + str(time2 - time1))

    # Load factors
    factor_data = read_factor_views(cur, factor_view, filt)
    time1 = time2
    time2 = time.time()
    print("read_factor_vews: " + str(time2 - time1))

    factor = np.zeros(len(factor_data), Factor)
    factor_pt = np.zeros(len(factor_data), np.str_)  # partition type
    factor_pid = np.zeros(len(factor_data), np.int64)  # partition id
    time1 = time2
    time2 = time.time()
    print("allocate: " + str(time2 - time1))

    for (i, f) in enumerate(factor_data):
        factor[i]["factorFunction"] = f[4]
        factor[i]["weightId"] = f[1]
        factor[i]["featureValue"] = f[2]
        factor[i]["arity"] = len(f[0])
        factor_pt[i] = f[3][0]
        # TODO: is factor_pid actually used anywhere?
        factor_pid[i] = -1
        if f[3][1:] != "":
            factor_pid[i] = int(f[3][1:])
    time1 = time2
    time2 = time.time()
    print("for loop: " + str(time2 - time1))

    if len(factor) > 0:
        factor[0]["ftv_offset"] = 0
    for i in range(1, len(factor)):
        factor[i]["ftv_offset"] = factor[i - 1]["ftv_offset"] \
                                + factor[i - 1]["arity"]
    time1 = time2
    time2 = time.time()
    print("ftv_offset for loop: " + str(time2 - time1))

    edges = 0
    if len(factor) > 0:
        edges = factor[-1]["ftv_offset"] + factor[-1]["arity"]
    fmap = np.zeros(edges, FactorToVar)
    time1 = time2
    time2 = time.time()
    print("fmap allocate: " + str(time2 - time1))

    index = 0
    for i in factor_data:
        for j in i[0]:
            fmap[index]["vid"] = j
            # TODO: how to actually get categorical info?
            fmap[index]["dense_equal_to"] = 0
            index += 1
    time1 = time2
    time2 = time.time()
    print("fmap setting: " + str(time2 - time1))
    print()
    print(all(factor == factor2))
    print(all(factor_pt == factor_pt2))
    print(all(fmap == fmap2))
    print(fmap.size)
    print(fmap2.size)
    print()

    # Load variable info
    var_data = read_views(cur, variable_view, filt)
    time1 = time2
    time2 = time.time()
    print("read_views var: " + str(time2 - time1))

    vid = np.zeros(len(var_data), np.int64)
    variable = np.zeros(len(var_data), Variable)
    var_pt = np.zeros(len(var_data), np.str_)  # partition type
    var_pid = np.zeros(len(var_data), np.int64)  # partition id
    for (i, v) in enumerate(var_data):
        vid[i] = v[0]
        variable[i]["isEvidence"] = v[1]
        variable[i]["initialValue"] = v[2]
        variable[i]["dataType"] = v[3]
        variable[i]["cardinality"] = v[4]
        # variable[i]["vtf_offset"] = ???
        var_pt[i] = v[5][0]
        var_pid[i] = -1
        if v[5][1:] != "":
            var_pid[i] = int(v[5][1:])
    time1 = time2
    time2 = time.time()
    print("vars for loop: " + str(time2 - time1))

    print(str(len(vid)) + " variables")
    perm = vid.argsort()
    vid = vid[perm]
    variable = variable[perm]
    var_pt = var_pt[perm]
    var_pid = var_pid[perm]
    time1 = time2
    time2 = time.time()
    print("sorting vars: " + str(time2 - time1))
    (vid_, variable_, var_pt_, var_pid_) = get_variables(cur, variable_view, filt)
    time1 = time2
    time2 = time.time()
    print("get_variables: " + str(time2 - time1))

    # remap factor to variable
    remap_fmap(fmap, vid)

    time1 = time2
    time2 = time.time()
    print("remap fmap: " + str(time2 - time1))

    # Load weight info
    # No filter since weights do not have a partition id
    weight_data = read_views(cur, weight_view, True)

    time1 = time2
    time2 = time.time()
    print("read weight: " + str(time2 - time1))

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

    time1 = time2
    time2 = time.time()
    print("weight for loop: " + str(time2 - time1))

    domain_mask = np.full(len(variable), True, np.bool)

    time1 = time2
    time2 = time.time()
    print("allocate domain_mask: " + str(time2 - time1))

    return (weight, variable, factor, fmap, domain_mask, edges, var_pt,
            var_pid, factor_pt, factor_pid, vid)


def serialize(array):
    """TODO."""
    try:
        return array.tobytes().decode('utf16').encode('utf8')
    except:
        return array.tobytes()


def deserialize(array, dtype):
    """TODO."""
    try:
        ar = array.decode('utf8').encode('utf16').lstrip(codecs.BOM_UTF16)
        return np.fromstring(ar, dtype=dtype)
    except:
        return np.fromstring(array, dtype=dtype)
