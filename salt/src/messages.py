"""TODO."""

from __future__ import print_function
import numbskull
from numbskull.numbskulltypes import *
import numbskull.inference
import numpy as np
import codecs
import numba
import time
import networkx as nx
import nxmetis

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
def get_factors_helper(row, ff, factor, factor_pt, factor_ufo, fmap,
                       factor_index, fmap_index):
    """TODO."""
    for i in row:
        factor[factor_index]["factorFunction"] = ff
        factor[factor_index]["weightId"] = i[-4]
        factor[factor_index]["featureValue"] = i[-3]
        factor_pt[factor_index] = i[-2]
        factor_ufo[factor_index] = (i[-1] == 117)  # 117 == 'u'
        factor[factor_index]["arity"] = len(i) - 4
        if factor_index == 0:
            factor[factor_index]["ftv_offset"] = 0
        else:
            factor[factor_index]["ftv_offset"] = \
                factor[factor_index - 1]["ftv_offset"] + \
                factor[factor_index - 1]["arity"]
        factor_index += 1

        for j in i[:-4]:
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
    factor_ufo = np.zeros(factors, np.bool)  # unary factor optimization
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
        cmd = (", ".join(['"' + i + '"' for i in name[:-1]]) +
               ", ASCII(LEFT(partition_key, 1))" +  # partition key
               ", ASCII(SUBSTR(partition_key, 2, 1))")  # unary factor opt

        op = op_template.format(cmd=cmd, table_name=v, filter=sql_filter)
        cur.execute(op)
        while True:
            row = cur.fetchmany(10000)
            if row == []:
                break
            (factor_index, fmap_index) = \
                get_factors_helper(row, ff, factor, factor_pt, factor_ufo,
                                   fmap, factor_index, fmap_index)

    return factor, factor_pt.view('c'), factor_ufo, fmap, edges


@numba.jit(nopython=True, cache=True, nogil=True)
def get_variables_helper(row, vid, variable, var_pt, var_ufo, index):
    """TODO."""
    for v in row:
        vid[index] = v[0]
        variable[index]["isEvidence"] = v[1]
        variable[index]["initialValue"] = v[2]
        variable[index]["dataType"] = v[3]
        variable[index]["cardinality"] = v[4]
        var_pt[index] = v[5]
        var_ufo[index] = (v[6] == 117)  # 117 == 'u'
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
        op = op_template.format(cmd="COUNT(*)", table_name=v,
                                filter=sql_filter)
        cur.execute(op)
        n += cur.fetchone()[0]  # number of factors in this table

    vid = np.zeros(n, np.int64)
    variable = np.zeros(n, Variable)
    var_pt = np.zeros(n, np.int8)  # partition type
    var_ufo = np.zeros(n, np.bool)  # unary factor opt

    index = 0
    for v in views:
        cmd = ("vid, variable_role, init_value, variable_type, cardinality, " +
               "ASCII(LEFT(partition_key, 1)), " +  # partition key
               "ASCII(SUBSTR(partition_key, 2, 1))")  # unary factor opt
        op = op_template.format(cmd=cmd, table_name=v, filter=sql_filter)
        cur.execute(op)
        while True:
            row = cur.fetchmany(10000)
            if row == []:
                break
            index = get_variables_helper(row, vid, variable,
                                         var_pt, var_ufo, index)

    perm = vid.argsort()
    vid = vid[perm]
    variable = variable[perm]
    var_pt = var_pt[perm]
    var_ufo = var_ufo[perm]

    return vid, variable, var_pt.view('c'), var_ufo


@numba.jit(nopython=True, cache=True, nogil=True)
def get_weights_helper(row, weight):
    """TODO."""
    for w in row:
        wid = w[0]
        weight[wid]["isFixed"] = w[1]
        weight[wid]["initialValue"] = w[2]


def get_weights(cur, views, sql_filter="True"):
    """TODO."""
    op_template = "SELECT {cmd} FROM {table_name} " \
                  "WHERE {filter}"

    # Obtain count of variables
    # TODO: is there a way to do this together with next part?
    #       (one query per table)
    n = 0
    for v in views:
        op = op_template.format(cmd="COUNT(*)", table_name=v,
                                filter=sql_filter)
        cur.execute(op)
        n += cur.fetchone()[0]  # number of factors in this table

    weight = np.zeros(n, Weight)

    index = 0
    for v in views:
        op = op_template.format(cmd="*", table_name=v, filter=sql_filter)
        cur.execute(op)
        while True:
            row = cur.fetchmany(10000)
            if row == []:
                break
            index = get_weights_helper(row, weight)

    return weight


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
    print(index)
    print(forward[ans])
    assert(forward[ans] == index)
    return ans


@numba.jit(nopython=True, cache=True, nogil=True)
def remap_fmap(fmap, vid):
    """TODO."""
    for i in range(len(fmap)):
        fmap[i]["vid"] = inverse_map(vid, fmap[i]["vid"])


def get_fg_data(cur, filt):
    """TODO."""
    print("***GET_FG_DATA***")

    # Get names of views
    time1 = time.time()
    (factor_view, variable_view, weight_view) = get_views(cur)
    time2 = time.time()
    print("get_views: " + str(time2 - time1))

    # Load factors
    (factor, factor_pt, factor_ufo, fmap, edges) = get_factors(cur, factor_view, filt)
    time1 = time2
    time2 = time.time()
    print("get_factors: " + str(time2 - time1))

    # Load variables
    (vid, variable, var_pt, var_ufo) = get_variables(cur, variable_view, filt)
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
    weight = get_weights(cur, weight_view)
    time1 = time2
    time2 = time.time()
    print("get_weight: " + str(time2 - time1))

    domain_mask = np.full(len(variable), True, np.bool)
    time1 = time2
    time2 = time.time()
    print("allocate domain_mask: " + str(time2 - time1))

    return (weight, variable, factor, fmap, domain_mask, edges, var_pt,
            factor_pt, var_ufo, factor_ufo, vid)


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


def find_connected_components(conn, cur):
    """TODO."""
    # Open a cursor to perform database operations
    (factor_view, variable_view, weight_view) = get_views(cur)
    (factor, factor_pt, factor_ufo, fmap, edges) = get_factors(cur, factor_view)

    hyperedges = []
    for f in factor:
        newedge = []
        for i in range(f['ftv_offset'], f['ftv_offset'] + f['arity']):
            newedge.append(fmap[i]['vid'])
        hyperedges.append(newedge)
    G = nx.Graph()
    for e in hyperedges:
        for i in range(len(e)):
            for j in range(i + 1, len(e)):
                newedge = (e[i], e[j])
                G.add_edge(*e)

    cc = nx.connected_components(G)
    try:
        cur.execute("CREATE TABLE variable_to_cc "
                    "(dd_id bigint, cc_id bigint);")
    except:
        conn.rollback()
        cur.execute("TRUNCATE variable_to_cc;")

    rows = []
    cc_id = 0
    for c in cc:
        for node in c:
            rows.append([node, cc_id])
        cc_id += 1

    dataText = ','.join(cur.mogrify('(%s,%s)', row) for row in rows)
    try:
        cur.execute("INSERT INTO variable_to_cc VALUES " + dataText)
        if cc_id > 1:
            cur.execute("CREATE INDEX dd_cc ON variable_to_cc (dd_id);")
        conn.commit()
        G.clear()
        return True
    except:
        conn.rollback()
        G.clear()
        return False

def find_metis_parts(conn, cur, parts):
    """TODO"""
    # Open a cursor to perform database operations
    (factor_view, variable_view, weight_view) = get_views(cur)
    # Obtain graph
    (factor, factor_pt, factor_ufo, fmap, edges) = get_factors(cur, factor_view)

    hyperedges = []
    for f in factor:
        newedge = []
        for i in range(f['ftv_offset'], f['ftv_offset']+f['arity']):
            newedge.append(fmap[i]['vid'])
        hyperedges.append(newedge)
    G = nx.Graph()
    for e in hyperedges:
        for i in range(len(e)):
            for j in range(i+1, len(e)):
                newedge = (e[i],e[j])
                G.add_edge(*e)
    # Run metis to obtain partitioning
    metis_options = nxmetis.MetisOptions(objtype=nxmetis.enums.MetisObjType.vol)
    cost, partitions = nxmetis.partition(G, parts, options=metis_options)
    print(80 * "*")
    print(cost)
    print(partitions)
    print(80 * "*")
    
    # Find nodes to master
    master_variables = set([])
    # Get all edges
    cut_edges = set(G.edges())
    for p in partitions:
        H = G.subgraph(p)
        cut_edges -= set(H.edges())
        print(H.edges())
        H.clear()
    for edge in cut_edges:
        n1, n2 = edge
        master_variables.add(n1)
        master_variables.add(n2)
    # Store parition in DB
    try:
        cur.execute("CREATE TABLE variable_to_cc (dd_id bigint, cc_id bigint);")
    except:
        conn.rollback()
        cur.execute("TRUNCATE variable_to_cc;")

    rows = []
    # Output master variables
    for node in master_variables:
        rows.append([node, -1])

    print(master_variables)
    # Output minion variables
    pid = 0
    for p in partitions:
        only_master = True
        for node in p:
            if node not in master_variables:
                only_master = False
                rows.append([node, pid])
        if not only_master:
            pid += 1
    print(rows)
    dataText = ','.join(cur.mogrify('(%s,%s)', row) for row in rows)
    print(dataText)
    try:
        cur.execute("INSERT INTO variable_to_cc VALUES " + dataText)
        if pid > 1:
            cur.execute("CREATE INDEX dd_cc ON variable_to_cc (dd_id);")
        conn.commit()
        G.clear()
        return True
    except:
        conn.rollback()
        G.clear()
        return False    
