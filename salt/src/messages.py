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
def get_factors_helper(row, fid, ff, factor, factor_pt, factor_ufo, fmap,
                       factor_index, fmap_index):
    """TODO."""
    for i in row:
        fid[factor_index] = i[-1]
        factor[factor_index]["factorFunction"] = ff
        factor[factor_index]["weightId"] = i[-5]
        factor[factor_index]["featureValue"] = i[-4]
        factor_pt[factor_index] = i[-3]
        factor_ufo[factor_index] = (i[-2] == 117)  # 117 == 'u'
        factor[factor_index]["arity"] = len(i) - 5
        if factor_index == 0:
            factor[factor_index]["ftv_offset"] = 0
        else:
            factor[factor_index]["ftv_offset"] = \
                factor[factor_index - 1]["ftv_offset"] + \
                factor[factor_index - 1]["arity"]
        factor_index += 1

        for j in i[:-5]:
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
    min_fid = np.zeros(len(views), np.int64)
    max_fid = np.zeros(len(views), np.int64)
    for (i, table) in enumerate(views):
        op = op_template.format(cmd="COUNT(*), MIN(fid), MAX(fid)", table_name=table,
                                filter=sql_filter)
        cur.execute(op)
        info = cur.fetchone()
        f = info[0]  # number of factors in this table
        min_fid[i] = info[1] if info[1] is not None else 0
        max_fid[i] = info[2] if info[2] is not None else 0

        count = count_template.format(table_name=table)
        cur.execute(count)
        v = cur.fetchone()[0] - 4  # number of vars used by these factors

        factors += f
        edges += f * v
    perm = min_fid.argsort()
    min_fid = min_fid[perm]
    max_fid = max_fid[perm]
    assert(all(max_fid[i] <= max_fid[i + 1] for i in xrange(len(max_fid) - 1)))
    # TODO: cannot directly apply perm to views (standard array, not numpy array)
    views_temp = [None for i in range(len(views))]
    for i in range(len(views)):
        views_temp[i] = views[perm[i]]
    views = views_temp

    fid = np.zeros(factors, np.int64)
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
        assert(name[-4] == "weight_id")
        assert(name[-3] == "feature_value")
        assert(name[-2] == "partition_key")
        assert(name[-1] == "fid")
        cmd = (", ".join(['"' + i + '"' for i in name[:-2]]) +
               ", ASCII(LEFT(partition_key, 1))" +  # partition key
               ", ASCII(SUBSTR(partition_key, 2, 1))" +  # unary factor opt
               ", fid")

        # TODO: should actually put the ORDER BY fid in its own var
        op = op_template.format(cmd=cmd, table_name=v, filter=sql_filter + "ORDER BY fid")
        cur.execute(op)
        while True:
            row = cur.fetchmany(10000)
            if row == []:
                break
            (factor_index, fmap_index) = \
                get_factors_helper(row, fid, ff, factor, factor_pt, factor_ufo,
                                   fmap, factor_index, fmap_index)

    return fid, factor, factor_pt.view('c'), factor_ufo, fmap, edges


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
    assert(forward[ans] == index)
    return ans

@numba.jit(nopython=True, cache=True, nogil=True)
def variable_exists(forward, index):
    """TODO."""
    ans = np.searchsorted(forward, index)
    return ans < len(forward) and forward[ans] == index


@numba.jit(nopython=True, cache=True, nogil=True)
def remap_fmap(fmap, vid):
    """TODO."""
    for i in range(len(fmap)):
        fmap[i]["vid"] = inverse_map(vid, fmap[i]["vid"])


@numba.jit(nopython=True, cache=True, nogil=True)
def remap_ufo(ufo, vid):
    """TODO."""
    for i in range(len(ufo)):
        ufo[i]["vid"] = inverse_map(vid, ufo[i]["vid"])


def get_fg_data(cur, filt, ismaster):
    """TODO."""
    print("***GET_FG_DATA***")

    # Get names of views
    time1 = time.time()
    (factor_view, variable_view, weight_view) = get_views(cur)
    time2 = time.time()
    print("get_views: " + str(time2 - time1))

    # Load factors
    (fid, factor, factor_pt, factor_ufo, fmap, edges) = get_factors(cur, factor_view, filt)
    time1 = time2
    time2 = time.time()
    print("get_factors: " + str(time2 - time1))

    # Load variables
    (vid, variable, var_pt, var_ufo) = get_variables(cur, variable_view, filt)
    time1 = time2
    time2 = time.time()
    print("get_variables: " + str(time2 - time1))

    print("factor: ", factor)
    print("factor_pt: ", factor_pt)
    print("factor_ufo: ", factor_ufo)
    print("fmap: ", fmap)
    print("edges: ", edges)
    print("vid: ", vid)
    print("variable: ", variable)
    print("var_pt: ", var_pt)
    print("var_ufo: ", var_ufo)
    print()

    fmap, vid, variable, var_pt, var_ufo, pf_list, pf_var_begin, pf_ufo_var_list = process_pf(factor, factor_pt, factor_ufo, fmap, fid, vid, variable, var_pt, var_ufo, ismaster)
    time1 = time2
    time2 = time.time()
    print("process_pf: " + str(time2 - time1))

    print("factor: ", factor)
    print("factor_pt: ", factor_pt)
    print("factor_ufo: ", factor_ufo)
    print("fmap: ", fmap)
    print("edges: ", edges)
    print("vid: ", vid)
    print("variable: ", variable)
    print("var_pt: ", var_pt)
    print("var_ufo: ", var_ufo)
    print()
    print("pf_list: ", pf_list)
    print("pf_var_begin: ", pf_var_begin)
    print("pf_ufo_var_list: ", pf_ufo_var_list)

    factor, factor_pt, factor_ufo, fmap, vid, variable, var_pt, var_ufo, ufo_send, ufo_recv, ufo_start, ufo_map, ufo_var_begin = process_ufo(factor, factor_pt, factor_ufo, fmap, vid, variable, var_pt, var_ufo, pf_ufo_var_list, pf_var_begin)
    time1 = time2
    time2 = time.time()
    print("process_ufo: " + str(time2 - time1))

    print("factor: ", factor)
    print("factor_pt: ", factor_pt)
    print("factor_ufo: ", factor_ufo)
    print("fmap: ", fmap)
    print("edges: ", edges)
    print("vid: ", vid)
    print("variable: ", variable)
    print("var_pt: ", var_pt)
    print("var_ufo: ", var_ufo)
    print()

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

    factors_to_skip = compute_skipped_factors(factor, factor_pt.view(np.int8), factor_ufo, fmap, fid, vid, variable, var_pt.view(np.int8), var_ufo)

    return (weight, variable, factor, fmap, domain_mask, edges, var_pt,
            factor_pt, var_ufo, factor_ufo, fid, vid, ufo_send, ufo_recv, ufo_start, ufo_map, ufo_var_begin, pf_list, factors_to_skip)


@numba.jit(nopython=True, cache=True, nogil=True)
def compute_skipped_factors(factor, factor_pt, factor_ufo, fmap, fid, vid, variable, var_pt, var_ufo):
    return np.zeros(0, np.int64)


def serialize(array):
    """TODO."""
    return array.tolist()
    # try:
    #     return array.tobytes().decode('utf16').encode('utf8')
    # except:
    #     return array.tobytes()


def deserialize(array, dtype):
    """TODO."""
    try:
        return np.array(array, dtype=dtype)
    except:
        # For UnaryFactorOpt and other complicated dtypes
        # Salt converts list of tuples into list of lists, which breaks the original version
        return np.array([tuple(i) for i in array], dtype=dtype)
    # try:
    #     ar = array.decode('utf8').encode('utf16').lstrip(codecs.BOM_UTF16)
    #     return np.fromstring(ar, dtype=dtype)
    # except:
    #     return np.fromstring(array, dtype=dtype)


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

@numba.jit(cache=True, nogil=True)
def remove_noop(factor, factor_pt, factor_ufo, fmap):

    factor_des, fmap_des = remove_noop_helper(factor, factor_pt, factor_ufo, fmap)

    factor = np.resize(factor, factor_des)
    factor_pt = np.resize(factor_pt, factor_des)
    factor_ufo = np.resize(factor_ufo, factor_des)

    fmap = np.resize(fmap, fmap_des)

    return factor, factor_pt, factor_ufo, fmap, fmap_des


@numba.jit(nopython=True, cache=True, nogil=True)
def remove_noop_helper(factor, factor_pt, factor_ufo, fmap):
    factor_des = 0
    fmap_des = 0
    ftv_offset = 0

    for factor_src in range(len(factor)):
        if factor[factor_src]["factorFunction"] == numbskull.inference.FUNC_NOOP:
            continue

        factor[factor_des] = factor[factor_src]
        factor_pt[factor_des] = factor_pt[factor_src]
        factor_ufo[factor_des] = factor_ufo[factor_src]

        factor[factor_des]["ftv_offset"] = ftv_offset
        ftv_offset += factor[factor_des]["arity"]

        for i in range(factor[factor_src]["arity"]):
            fmap[fmap_des + i] = fmap[factor[factor_src]["ftv_offset"] + i]

        fmap_des += factor[factor_des]["arity"]
        factor_des += 1

    return factor_des, fmap_des


@numba.jit(nopython=True, cache=True, nogil=True)
def find_ufo(factor, factor_pt, factor_ufo, fmap, vid, variable, var_pt, var_ufo, pf_ufo_var_list, pf_var_begin):
    # Count number of factors with UFO
    n_ufo_recv = 0  # Number of ufo to receive
    n_ufo_send = 0  # Number of ufo to send
    for i in range(len(factor)):
        if factor_ufo[i]:
            exist = 0  # number of vars manifested on this machine
            for j in range(factor[i]["arity"]):
                vid1 = fmap[factor[i]["ftv_offset"] + j]["vid"]
                local_vid = loose_inverse_map(vid, vid1)
                exist += (local_vid != -1) and (var_pt[local_vid] != 80 or var_ufo[local_vid])
                # (local_vid != -1) specifies that this var must be on this machine to exist
                # (var_pt[local_vid] != 80 or var_ufo[local_vid])
                # part 1 (check against 80) mean that this is not a partial factor var
                # part 2 is a check that it replaced an ufo var

            # Must have exactly one or all vars on this machine
            assert(exist == 1 or exist == factor[i]["arity"])

            if exist == 1:
                # One var is missing
                # This machine gets the UFO
                n_ufo_recv += 1
            else:
                # All vars are present
                # This machine computes the UFO
                n_ufo_send += 1

    ufo_recv = np.empty(n_ufo_recv, dtype=UnaryFactorOpt)
    ufo_send = np.empty(n_ufo_send, dtype=UnaryFactorOpt)
    n_ufo_recv = 0
    n_ufo_send = 0
    for i in range(len(factor)):
        if factor_ufo[i]:
            exist = 0  # number of vars manifested on this machine
            var = -1
            is_pf = False
            for j in range(factor[i]["arity"]):
                vid1 = fmap[factor[i]["ftv_offset"] + j]["vid"]
                local_vid = loose_inverse_map(vid, vid1)
                ex = (local_vid != -1) and (var_pt[local_vid] != 80 or var_ufo[local_vid])
                exist += ex
                if ex:
                    var = vid1
                if (local_vid != -1) and var_pt[local_vid] == 80:
                    is_pf = True

            if exist == 1:
                # Only one var on this machine
                # This machine receives the ufo
                ufo_recv[n_ufo_recv]['vid'] = var
                ufo_recv[n_ufo_recv]['weightId'] = factor[i]['weightId']

                n_ufo_recv += 1
                if not is_pf:
                    factor[i]["factorFunction"] = numbskull.inference.FUNC_NOOP
            else:
                # Both on this machine
                # Check which is actually the UFO
                var = -1
                for j in range(factor[i]["arity"]):
                    vid1 = fmap[factor[i]["ftv_offset"] + j]["vid"]
                    local_vid = inverse_map(vid, vid1)
                    is_ufo = var_ufo[local_vid]
                    if is_ufo:
                        assert(var == -1)  # This must be the first seen
                        is_pf = (var_pt[local_vid] == 80)  # check if this is a partial factor
                        if is_pf:
                            var = pf_ufo_var_list[local_vid - pf_var_begin]
                        else:
                            var = vid1
                # if var == -1:
                #     # no ufo var found yet
                #     # this factor must have been partial factored
                #     # last var has to be the partial factor var
                #     vid1 = fmap[factor[i]["ftv_offset"] + factor[i]["arity"] - 1]["vid"]
                #     local_vid = inverse_map(vid, vid1)
                #     is_pf = (var_pt[local_vid] == 80)  # check that this is a partial factor
                #     assert(is_pf)
                #     var = pf_ufo_var_list[local_vid - pf_var_begin]

                ufo_send[n_ufo_send]['vid'] = var
                ufo_send[n_ufo_send]['weightId'] = factor[i]['weightId']

                n_ufo_send += 1

    return ufo_send, ufo_recv

@numba.jit(nopython=True, cache=True, nogil=True)
def extra_space(vid, variable, ufo_recv):
    m_factors = len(ufo_recv)
    m_fmap = 0
    m_var = 0
    for ufo in ufo_recv:
        card = variable[ufo["vid"]]["cardinality"]
        m_fmap += card
        m_var += card - 1
    return m_factors, m_fmap, m_var

@numba.jit(nopython=True, cache=True, nogil=True)
def set_ufo(factor, factor_pt, factor_ufo, fmap, vid, variable, var_pt, var_ufo, ufo_recv, n_factors, n_fmap, n_var, vid_max):
    # vid_max should just be np.iinfo(vid.dtype).max, but numba doesn't support iinfo
    ftv_offset = 0
    if len(factor) > 0:
        ftv_offset = factor[-1]["ftv_offset"] + factor[-1]["arity"]

    n_vid = vid_max - len(vid) + n_var + 1
    for (i, ufo) in enumerate(ufo_recv):
        card = variable[ufo["vid"]]["cardinality"]

        factor[n_factors + i]["factorFunction"] = numbskull.inference.FUNC_UFO
        factor[n_factors + i]["weightId"] = ufo["weightId"]
        factor[n_factors + i]["featureValue"] = 1 # TODO: feature value may not match
        factor[n_factors + i]["arity"] = card
        factor[n_factors + i]["ftv_offset"] = ftv_offset

        factor_pt[n_factors + i] = 85  # TODO: Does this actually matter at all?

        factor_ufo[n_factors + i] = True

        fmap[n_fmap]["vid"] = vid[ufo["vid"]]
        n_fmap += 1
        for j in range(card - 1):
            fmap[n_fmap]["vid"] = n_vid

            vid[n_var] = n_vid

            variable[n_var]["isEvidence"] = 4
            variable[n_var]["initialValue"]
            variable[n_var]["dataType"]
            variable[n_var]["cardinality"]
            variable[n_var]["vtf_offset"]

            var_pt[n_var] = 85  # TODO: Does this actually matter at all?

            var_ufo[n_var] = True

            n_vid += 1
            n_fmap += 1
            n_var += 1

        ftv_offset += card


@numba.jit(cache=True, nogil=True)
def add_ufo(factor, factor_pt, factor_ufo, fmap, vid, variable, var_pt, var_ufo, ufo_recv, pf_var_begin):
    n_factors = len(factor)
    n_fmap = len(fmap)
    n_var = len(variable)

    m_factors, m_fmap, m_var = extra_space(vid, variable, ufo_recv)

    factor = np.resize(factor, n_factors + m_factors)
    factor_pt = np.resize(factor_pt, n_factors + m_factors)
    factor_ufo = np.resize(factor_ufo, n_factors + m_factors)

    fmap = np.resize(fmap, n_fmap + m_fmap)

    vid = np.resize(vid, n_var + m_var)
    variable = np.resize(variable, n_var + m_var)
    var_pt = np.resize(var_pt, n_var + m_var)
    var_ufo = np.resize(var_ufo, n_var + m_var)

    # need to decrease vids of pf's to not overlap with ufo fake vid
    decrease_vid(fmap, vid, m_var, pf_var_begin, n_var)

    set_ufo(factor, factor_pt, factor_ufo, fmap, vid, variable, var_pt, var_ufo, ufo_recv, n_factors, n_fmap, n_var, np.iinfo(vid.dtype).max)

    return factor, factor_pt, factor_ufo, fmap, vid, variable, var_pt, var_ufo, n_var


@numba.jit(nopython=True, cache=True, nogil=True)
def decrease_vid(fmap, vid, amount, begin, end):
    for i in range(len(fmap)):
        if vid[begin] <= fmap[i]["vid"] <= vid[end - 1]:
            fmap[i]["vid"] -= amount
    vid[begin:end] -= amount


@numba.jit(nopython=True, cache=True, nogil=True)
def ufo_equal(u, v):
    """Numba-compatible equality check."""

    # TODO: is there a way to do this in a safer way?
    #       (in the sense that this changes if def of UFO changes)
    return u["vid"] == v["vid"] and \
           u["weightId"] == v["weightId"]


@numba.jit(nopython=True, cache=True, nogil=True)
def ufo_less(u, v):
    """Numba-compatible equality check."""

    # TODO: is there a way to do this in a safer way?
    if u["vid"] != v["vid"]:
        return u["vid"] < v["vid"]
    return u["weightId"] < v["weightId"]


@numba.jit(nopython=True, cache=True, nogil=True)
def ufo_check_sorted(a):
    """Checks if a numpy-array of ufo's is sorted."""
    for i in range(1, len(a)):
        assert(ufo_less(a[i - 1], a[i]))


@numba.jit(nopython=True, cache=True, nogil=True)
def ufo_searchsorted(a, b):
    begin = -1
    end = len(a)
    while begin + 1 < end:
        mid = (begin + end) / 2
        if ufo_less(a[mid], b):
            begin = mid
        else:
            end = mid
    return end


@numba.jit(nopython=True, cache=True, nogil=True)
def compute_ufo_map(factor, factor_pt, factor_ufo, fmap, vid, variable, var_pt, var_ufo, ufo_send, pf_ufo_var_list, pf_var_begin):
    ufo_length = np.zeros(ufo_send.size, np.int64)
    ufo_start = np.zeros(ufo_send.size + 1, np.int64)
    ufo = np.zeros(1, dtype=UnaryFactorOpt)
    if len(ufo_send) == 0:
        return ufo_length, np.zeros(0, np.int64)

    for i in range(len(factor)):
        if factor_ufo[i]:
            exist = 0  # number of vars manifested on this machine
            var = -1
            for j in range(factor[i]["arity"]):
                vid1 = fmap[factor[i]["ftv_offset"] + j]["vid"]
                local_vid = inverse_map(vid, vid1)
                ex = (local_vid != -1) and (var_pt[local_vid] != 80 or var_ufo[local_vid])
                exist += ex
                if ex and var_ufo[local_vid]:
                    # This variable is on this machine and is ufo
                    assert(var == -1)  # This must be the first seen
                    is_pf = (var_pt[local_vid] == 80)  # check if this is a partial factor
                    if is_pf:
                        var = pf_ufo_var_list[local_vid - pf_var_begin]
                    else:
                        var = local_vid
            # if var == -1:
            #     # no ufo var found yet
            #     # this factor must have been partial factored
            #     # last var has to be the partial factor var
            #     vid1 = fmap[factor[i]["ftv_offset"] + factor[i]["arity"] - 1]["vid"]
            #     local_vid = inverse_map(vid, vid1)
            #     is_pf = (var_pt[local_vid] == 80)  # check that this is a partial factor
            #     assert(is_pf)
            #     var = pf_ufo_var_list[local_vid - pf_var_begin]

            # Must have exactly one or all vars on this machine
            assert(exist == 1 or exist == factor[i]["arity"])

            if exist == 1:
                # Only one var on this machine
                # This machine receives the ufo
                # No computation will be done
                pass
            else:
                # All vars on this machine
                # Will be computing
                weightId = factor[i]['weightId']

                # TODO: is there a way to not create a list of length 1
                ufo[0]["vid"] = var
                ufo[0]["weightId"] = weightId
                j = ufo_searchsorted(ufo_send, ufo[0])
                assert(ufo_equal(ufo_send[j], ufo[0]))

                ufo_length[j] += 1

    for i in range(ufo_send.size):
        ufo_start[i + 1] = ufo_start[i] + ufo_length[i]
        ufo_length[i] = 0

    ufo_map = np.zeros(ufo_start[-1], np.int64)

    for i in range(len(factor)):
        if factor_ufo[i]:
            exist = 0
            var = -1
            for j in range(factor[i]["arity"]):
                vid1 = fmap[factor[i]["ftv_offset"] + j]["vid"]
                local_vid = inverse_map(vid, vid1)
                ex = (local_vid != -1) and (var_pt[local_vid] != 80 or var_ufo[local_vid])
                exist += ex
                if ex and var_ufo[local_vid]:
                    # This variable is on this machine and is ufo
                    assert(var == -1)  # This must be the first seen
                    is_pf = (var_pt[local_vid] == 80)  # check if this is a partial factor
                    if is_pf:
                        var = pf_ufo_var_list[local_vid - pf_var_begin]
                    else:
                        var = local_vid
            # if var == -1:
            #     # no ufo var found yet
            #     # this factor must have been partial factored
            #     # last var has to be the partial factor var
            #     vid1 = fmap[factor[i]["ftv_offset"] + factor[i]["arity"] - 1]["vid"]
            #     local_vid = inverse_map(vid, vid1)
            #     is_pf = (var_pt[local_vid] == 80)  # check that this is a partial factor
            #     assert(is_pf)
            #     var = pf_ufo_var_list[local_vid - pf_var_begin]

            # Must have exactly one or all vars on this machine
            assert(exist == 1 or exist == factor[i]["arity"])

            if exist == factor[i]["arity"]:
                weightId = factor[i]['weightId']

                ufo[0]["vid"] = var
                ufo[0]["weightId"] = weightId
                j = ufo_searchsorted(ufo_send, ufo[0])
                assert(ufo_equal(ufo_send[j], ufo[0]))

                ufo_map[ufo_start[j] + ufo_length[j]] = i
                ufo_length[j] += 1

    return ufo_start, ufo_map


@numba.jit(nopython=True, cache=True, nogil=True)
def compute_ufo_values(factor, fmap, var_value, variable, var_ufo, ufo_send, ufo_start, ufo_map, ufo):
    var_copy = 0
    ufo_index = 0

    for i in range(len(ufo)):
        ufo[i] = 0

    for i in range(len(ufo_send)):
        var_samp = ufo_send[i]["vid"]
        for j in range(ufo_start[i], ufo_start[i + 1]):
            factor_id = ufo_map[j]
            value = 0

            f0 = numbskull.inference.eval_factor(factor_id, var_samp, value, var_copy, variable, factor, fmap, var_value)

            for value in range(1, variable[var_samp]["cardinality"]):
                ufo[ufo_index + value - 1] += numbskull.inference.eval_factor(factor_id, var_samp, value, var_copy, variable, factor, fmap, var_value) - f0
        ufo_index += variable[var_samp]["cardinality"] - 1


@numba.jit(nopython=True, cache=True, nogil=True)
def clear_ufo_values(var_value, ufo_var_begin):
    for i in range(ufo_var_begin, len(var_value)):
        var_value[i] = 0


@numba.jit(nopython=True, cache=True, nogil=True)
def apply_ufo_values(factor, fmap, var_value, ufo_map, ufo_values):
    for i in range(len(ufo_map)):
        assert(factor[i]["arity"] == 2)
        var_value[fmap[factor[i]["ftv_offset"] + 1]["vid"]] += ufo_values[i]


@numba.jit(cache=True, nogil=True)
def process_pf(factor, factor_pt, factor_ufo, fmap, fid, vid, variable, var_pt, var_ufo, ismaster):
    """Process partial factor."""

    pf_var_begin = len(vid)

    pf_list = find_pf(factor, factor_pt.view(np.int8), factor_ufo, fmap, fid, vid, variable, var_pt.view(np.int8), var_ufo)

    vid = np.resize(vid, len(vid) + len(pf_list))
    variable = np.resize(variable, len(variable) + len(pf_list))
    var_pt = np.resize(var_pt, len(var_pt) + len(pf_list))
    var_ufo = np.resize(var_ufo, len(var_ufo) + len(pf_list))

    pf_ufo_var_list = np.zeros(pf_list.size, np.int64)  # This is a list of one of the vars that was deleted from a partial factor
                                                        # In this case that this factor is also UFO'd, then this var is the UFO var
    ftv_offset = set_pf(factor, factor_pt.view(np.int8), factor_ufo, fmap, fid, vid, variable, var_pt.view(np.int8), var_ufo, pf_var_begin, np.iinfo(vid.dtype).max, ismaster, pf_ufo_var_list)

    fmap = np.resize(fmap, ftv_offset)

    return fmap, vid, variable, var_pt.view('c'), var_ufo, pf_list, pf_var_begin, pf_ufo_var_list


@numba.jit(nopython=True, cache=True, nogil=True)
def find_pf(factor, factor_pt, factor_ufo, fmap, fid, vid, variable, var_pt, var_ufo):
    count = 0
    for i in range(len(factor)):
        if ((factor_pt[i] == 68 and not factor_ufo[i])  # "D"
          or factor_pt[i] == 71):  # "G"
            count += 1

    pf_list = np.zeros(count, np.int64)
    count = 0
    for i in range(len(factor)):
        if ((factor_pt[i] == 68 and not factor_ufo[i])  # "D"
          or factor_pt[i] == 71):  # "G"
            pf_list[count] = i
            count += 1

    return pf_list


@numba.jit(nopython=True, cache=True, nogil=True)
def set_pf(factor, factor_pt, factor_ufo, fmap, fid, vid, variable, var_pt, var_ufo, pf_var_begin, vid_max, ismaster, pf_ufo_var_list):
    # vid_max should just be np.iinfo(vid.dtype).max, but numba doesn't support iinfo   

    # Setting fake variables
    for i in range(pf_var_begin, len(vid)):
        vid[i] = vid_max - len(vid) + i  # I think this can have a +1, but it doesn't really matter

        variable[i]["isEvidence"] = 4
        variable[i]["initialValue"]
        variable[i]["dataType"]
        variable[i]["cardinality"]
        variable[i]["vtf_offset"]

        var_pt[i] = 80  # TODO: Does this actually matter at all?

        var_ufo[i] = False

    ftv_offset = 0
    ftv_offset_src = 0
    count = 0
    for i in range(len(factor)):
        factor[i]["ftv_offset"] = ftv_offset

        if ((factor_pt[i] == 68 and not factor_ufo[i])  # "D"
          or factor_pt[i] == 71):  # "G"
            # Is a partial factor
            arity = 0
            var_was_ufo = False
            for j in range(factor[i]["arity"]):
                # Minions are loading Au vars right now for PPB with UFO
                # Because PPB partitions on factors, the var never appears on minion
                # but information about the minion needs to exist to figure out how
                # to handle the UFO
                local_vid = loose_inverse_map(vid, fmap[ftv_offset + j]["vid"])
                if local_vid != -1 and (ismaster or var_pt[local_vid] != 65):  # "A"
                    fmap[ftv_offset + arity] = fmap[ftv_offset_src + j]
                    arity += 1
                else:
                    # This variable is being deleted
                    # Could be a ufo var
                    # save for later use
                    pf_ufo_var_list[count] = fmap[ftv_offset_src + j]["vid"]
                    var_was_ufo = var_ufo[local_vid]
            assert(arity < factor[i]["arity"])  # there isn't space allocated for extra vid

            fmap[ftv_offset + arity]["vid"] = vid[pf_var_begin + count]
            var_ufo[pf_var_begin + count] = var_was_ufo
            count += 1
            arity += 1
            factor[i]["arity"] = arity
            ftv_offset += arity
            ftv_offset_src += factor[i]["arity"]
        else:
            for j in range(factor[i]["arity"]):
                local_vid = loose_inverse_map(vid, fmap[ftv_offset + j]["vid"])
                if local_vid != -1 and (ismaster or var_pt[local_vid] != 65):  # "A"
                    fmap[ftv_offset + j] = fmap[ftv_offset_src + j]
            ftv_offset += factor[i]["arity"]
            ftv_offset_src += factor[i]["arity"]

    return ftv_offset


@numba.jit(cache=True, nogil=True)
def process_ufo(factor, factor_pt, factor_ufo, fmap, vid, variable, var_pt, var_ufo, pf_ufo_var_list, pf_var_begin):

    time1 = time.time()
    ufo_send, ufo_recv = find_ufo(factor, factor_pt.view(np.int8), factor_ufo, fmap, vid, variable, var_pt.view(np.int8), var_ufo, pf_ufo_var_list, pf_var_begin)
    time2 = time.time()
    print("find_ufo took ", time2 - time1)

    factor, factor_pt, factor_ufo, fmap, edges = remove_noop(factor, factor_pt.view(np.int8), factor_ufo, fmap)
    time1 = time2
    time2 = time.time()
    print("remove_noop took ", time2 - time1)

    # compute unique
    ufo_send = np.unique(ufo_send)
    ufo_recv = np.unique(ufo_recv)
    ufo_send.sort()
    ufo_recv.sort()

    # Checking that numpy sort uses the same comparison
    ufo_check_sorted(ufo_send)
    ufo_check_sorted(ufo_recv)

    remap_ufo(ufo_send, vid)
    remap_ufo(ufo_recv, vid)
    apply_loose_inverse_map(vid, pf_ufo_var_list)

    time1 = time2
    time2 = time.time()
    print("unique + sort took ", time2 - time1)

    # add fake factors vars for UFO
    factor, factor_pt, factor_ufo, fmap, vid, variable, var_pt, var_ufo, ufo_var_begin = add_ufo(factor, factor_pt.view(np.int8), factor_ufo, fmap, vid, variable, var_pt.view(np.int8), var_ufo, ufo_recv, pf_var_begin)
    time1 = time2
    time2 = time.time()
    print("add_ufo took ", time2 - time1)

    # Provide a fast method of finding factors that need to be evaluated for UFO
    ufo_start, ufo_map = compute_ufo_map(factor, factor_pt, factor_ufo, fmap, vid, variable, var_pt, var_ufo, ufo_send, pf_ufo_var_list, pf_var_begin)
    time1 = time2
    time2 = time.time()
    print("compute_ufo_map took ", time2 - time1)

    return factor, factor_pt.view('c'), factor_ufo, fmap, vid, variable, var_pt.view('c'), var_ufo, ufo_send, ufo_recv, ufo_start, ufo_map, ufo_var_begin

@numba.jit(nopython=True, cache=True, nogil=True)
def compute_map_master(vid, var_pt):
    l = 0
    for i in range(len(var_pt)):
        if var_pt[i] == 66:  # 66 = "B"
            l += 1

    map_to_minions = np.zeros(l, np.int64)
    l = 0
    for i in range(len(var_pt)):
        if var_pt[i] == 66:  # 66 = "B"
            map_to_minions[l] = vid[i]
            l += 1

    return map_to_minions


@numba.jit(nopython=True, cache=True, nogil=True)
def compute_map_minion(vid, var_pt):
    l = 0
    for i in range(len(var_pt)):
        if var_pt[i] == 68:  # 68 = "D"
            l += 1

    map_to_master = np.zeros(l, np.int64)
    l = 0
    for i in range(len(var_pt)):
        if var_pt[i] == 68:  # 68 = "D"
            map_to_master[l] = vid[i]
            l += 1

    return map_to_master


@numba.jit(nopython=True, cache=True, nogil=True)
def apply_inverse_map(vid, array):
    for i in range(len(array)):
        array[i] = inverse_map(vid, array[i])

@numba.jit(nopython=True, cache=True, nogil=True)
def loose_inverse_map(forward, index):
    """TODO."""
    ans = np.searchsorted(forward, index)
    if ans >= len(forward) or forward[ans] != index:
        return -1
    return ans


@numba.jit(nopython=True, cache=True, nogil=True)
def apply_loose_inverse_map(vid, array):
    for i in range(len(array)):
        array[i] = loose_inverse_map(vid, array[i])


@numba.jit(nopython=True, cache=True, nogil=True)
def compute_vars_to_send(map, var_to_send, var_value):
    # TODO: handle multiple copies
    for (i, m) in enumerate(map):
        var_to_send[i] = var_value[m]


@numba.jit(nopython=True, cache=True, nogil=True)
def process_received_vars(map, var_recv, var_value):
    for (i, v) in enumerate(var_recv):
        m = map[i]
        var_value[m] = v


@numba.jit(nopython=True, cache=True, nogil=True)
def ufo_to_factor(ufo, ufo_map, n_factors):
    index = np.empty(ufo.size, np.int64)
    for i in range(len(ufo)):
        j = ufo_searchsorted(ufo_map, ufo[i])
        assert(ufo_equal(ufo_map[j], ufo[i]))
        index[i] = n_factors - len(ufo_map) + j
    return index

@numba.jit(nopython=True, cache=True, nogil=True)
def compute_pf_values(factor, fmap, var_value, variable, pf_list, pf):
    for i in range(len(pf_list)):
        assert(factor[pf_list[i]]["factorFunction"] in [numbskull.inference.FUNC_OR,
                                                        numbskull.inference.FUNC_AND,
                                                        numbskull.inference.FUNC_ISTRUE])

        factor[pf_list[i]]["arity"] -= 1
        factor_id = pf_list[i]
        var_samp = -1
        value = -1
        var_copy = 0
        pf[i] = numbskull.inference.eval_factor(factor_id, var_samp, value, var_copy, variable, factor, fmap, var_value)
        factor[pf_list[i]]["arity"] += 1


@numba.jit(nopython=True, cache=True, nogil=True)
def apply_pf_values(factor, fmap, var_value, variable, pf_list, pf_values):
    for i in range(len(pf_list)):
        if pf_list[i] != -1:
            fac = factor[pf_list[i]]
            var_value[fmap[fac["ftv_offset"] + fac["arity"] - 1]["vid"]] = pf_values[i]
