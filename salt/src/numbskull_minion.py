"""TODO."""

# Import python libs
from __future__ import print_function, absolute_import
import json
import logging
import sys
import os
import time
import argparse
import numpy as np
import codecs

import pydoc
import psycopg2
import urlparse
import numpy as np
import traceback

# Import salt libs
import salt.utils.event

# Import numbskull
m_opts = salt.config.minion_config(os.environ['SALT_CONFIG_DIR'] + '/minion')
sys.path.append(m_opts['extension_modules'] + '/modules')
import numbskull
from numbskull import numbskull
from numbskull.numbskulltypes import *
import messages

log = logging.getLogger(__name__)


class NumbskullMinion:
    """TODO."""

    def __init__(self):
        """TODO."""
        self.partitionId = None
        self.args = None
        self.ns = None

    def parse_args(self, argv):
        """TODO."""
        if argv is None:
            argv = sys.argv[1:]
        parser = argparse.ArgumentParser(
            description="Runs a Gibbs sampler",
            epilog="")
        # Add version to parser
        parser.add_argument("--version",
                            action='version',
                            version="%(prog)s 0.0",
                            help="print version number")
        # Add execution arguments to parser
        for arg, opts in numbskull.arguments:
            parser.add_argument(*arg, **opts)
        # Add flags to parser
        for arg, opts in numbskull.flags:
            parser.add_argument(*arg, **opts)
        # Initialize NumbSkull #
        args = parser.parse_args(argv)
        return args

    def init_numbskull(self, argv):
        """TODO."""
        self.args = self.parse_args(argv)
        self.ns = numbskull.NumbSkull(**vars(self.args))

    def loadFG(self, data):
        """TODO."""
        try:
            weight = np.fromstring(data['weight'], dtype=Weight)
            variable = np.fromstring(data['variable'], dtype=Variable)
            factor = messages.deserialize(data['factor'], Factor)
            fmap = np.fromstring(data['fmap'], dtype=FactorToVar)
            vmap = np.fromstring(data['vmap'], dtype=VarToFactor)
            factor_index = np.fromstring(data['factor_index'], dtype=np.int64)
        except:
            return 'FAILED LOADING', None
        try:
            self.ns.loadFactorGraphRaw(weight, variable, factor,
                                       fmap, vmap, factor_index)
            fg = self.ns.factorGraphs[-1]
            meta = {}
            meta['weights'] = fg.weight.shape[0]
            meta['variables'] = fg.variable.shape[0]
            meta['factors'] = fg.factor.shape[0]
            return 'SUCCESS', meta
        except:
            return 'FAILED', None

    def learning(self, fgID):
        """TODO."""
        try:
            self.ns.learning(fgID, False)
            weights = self.ns.factorGraphs[fgID].weight_value
            return 'SUCCESS', messages.serialize(weights)
        except:
            return 'FAILED', None

    def inference(self, fgID=0):
        """TODO."""
        try:
            self.ns.inference(fgID, False)
            marginals = self.ns.factorGraphs[fgID].marginals
            return 'SUCCESS', messages.serialize(marginals)
        except:
            return 'FAILED', None


def start():
    """TODO."""
    log.debug('Initializing Numbskull Minion Engine')
    ns_minion = NumbskullMinion()
    event_bus = salt.utils.event.get_event(
            'minion',
            transport=__opts__['transport'],
            opts=__opts__,
            sock_dir=__opts__['sock_dir'],
            listen=True)
    log.debug('Starting Numbskull Minion Engine')
    partition_id = -1
    for evdata in event_bus.iter_events(full=True):
        loop_begin = time.time()
        tag, data = evdata['tag'], evdata['data']

        if tag == messages.ASSIGN_ID:
            partition_id = data['id']
            print("Assigned partition id #", partition_id)
            # TODO: respond to master
        elif tag == messages.INIT_NS:
            try:
                ns_minion.init_numbskull(data['argv'])
                # Respond OK to master
                data = {'status': 'OK'}
                __salt__['event.send'](messages.INIT_NS_RES, data)
            except:
                # Respond FAIL to master
                data = {'status': 'FAIL'}
                __salt__['event.send'](messages.INIT_NS_RES, data)
        elif tag == messages.LOAD_FG:
            # Connect to an existing database
            # http://stackoverflow.com/questions/15634092/connect-to-an-uri-in-postgres
            db_url = data["db_url"]
            url = urlparse.urlparse(db_url)
            username = url.username
            password = url.password
            database = url.path[1:]
            hostname = url.hostname
            port = url.port
            conn = psycopg2.connect(
                database=database,
                user=username,
                password=password,
                host=hostname,
                port=port
            )

            # Open a cursor to perform database operations
            cur = conn.cursor()
            minion_filter = "   partition_key similar to 'B(|u)' " \
                            "or partition_key similar to 'C(|u){partition_id}' " \
                            "or partition_key similar to 'D(|u){partition_id}' " \
                            "or partition_key similar to 'E(|u){partition_id}' " \
                            "or partition_key similar to 'F(|u){partition_id}' " \
                            "or partition_key similar to 'G(|u){partition_id}' " \
                            "or partition_key similar to 'H(|u)' "
            minion_filter = minion_filter.format(partition_id=partition_id)

            (weight, variable, factor, fmap, domain_mask, edges, var_pt,
             factor_pt, var_ufo, factor_ufo, fid, vid, ufo_send, ufo_recv, ufo_start, ufo_map, ufo_var_begin, pf_list) = \
                messages.get_fg_data(cur, minion_filter)

            # Close communication with the database
            cur.close()
            conn.close()

            variable[var_pt == "B"]["isEvidence"] = 4  # not owned var type

            ns_minion.ns.loadFactorGraph(weight, variable, factor, fmap,
                                         domain_mask, edges)

            # Respond to master
            data = {}
            __salt__['event.send'](messages.LOAD_FG_RES, data)
            log.debug("DONE LOADFG")
        elif tag == messages.SYNC_MAPPING:
            # receive map from master
            map_from_master = messages.deserialize(data["map"], np.int64)
            pf_from_master = messages.deserialize(data["pf"], np.int64)
            messages.apply_loose_inverse_map(fid, pf_from_master)

            # compute map
            map_to_master = messages.compute_map_minion(vid, var_pt.view(np.int8))

            ufo_to_master = ufo_send.copy()
            ufo_to_master["vid"] = vid[ufo_to_master["vid"]]
            data = {"pid": partition_id,
                    "map": messages.serialize(map_to_master),
                    "pf": messages.serialize(fid[pf_list]),
                    "ufo": messages.serialize(ufo_to_master)}
            __salt__['event.send'](messages.SYNC_MAPPING_RES, data)

            messages.apply_inverse_map(vid, map_from_master)
            messages.apply_inverse_map(vid, map_to_master)

            variables_to_master = np.zeros(map_to_master.size, np.int64)
            var_evid_to_master = np.zeros(map_to_master.size, np.int64)

            pf_to_master = np.zeros(pf_list.size, np.int64)
            pf_evid_to_master = np.zeros(pf_list.size, np.int64)

            m_factors, m_fmap, m_var = messages.extra_space(vid, variable, ufo_send)
            ufo_to_master = np.empty(m_var, np.int64)
            ufo_evid_to_master = np.empty(m_var, np.int64)

            log.debug("DONE SYNC_MAPPING")
        elif tag == messages.INFER or tag == messages.LEARN:
            variables_from_master = \
                messages.deserialize(data["values"], np.int64)
            messages.process_received_vars(map_from_master, variables_from_master, ns_minion.ns.factorGraphs[-1].var_value[0])
            messages.apply_pf_values(factor, fmap, ns_minion.ns.factorGraphs[-1].var_value[0], variable, pf_from_master, messages.deserialize(data["pf"], np.int64))

            if tag == messages.LEARN:
                var_evid_from_master = \
                    messages.deserialize(data["v_evid"], np.int64)
                messages.process_received_vars(map_from_master, var_evid_from_master, ns_minion.ns.factorGraphs[-1].var_value_evid[0])
                messages.apply_pf_values(factor, fmap, ns_minion.ns.factorGraphs[-1].var_value_evid[0], variable, pf_from_master, messages.deserialize(data["pf_evid"], np.int64))

                ns_minion.ns.factorGraphs[-1].weight_value[0] = \
                        messages.deserialize(data["weight"], np.float64)
                w0 = ns_minion.ns.factorGraphs[-1].weight_value[0]

            begin = time.time()

            fgID = 0
            if tag == messages.LEARN:
                ns_minion.ns.learning(fgID, False)
            else:
                ns_minion.ns.inference(fgID, False)

            end = time.time()
            log.debug("INFERENCE LOOP TOOK " + str(end - begin))

            # Respond to master
            messages.compute_vars_to_send(map_to_master, variables_to_master, ns_minion.ns.factorGraphs[-1].var_value[0])
            messages.compute_pf_values(factor, fmap, ns_minion.ns.factorGraphs[-1].var_value, variable, pf_list, pf_to_master)
            messages.compute_ufo_values(factor, fmap, ns_minion.ns.factorGraphs[-1].var_value, variable, var_ufo, ufo_send, ufo_start, ufo_map, ufo_to_master)
            print(80 * "*")
            print(ns_minion.ns.factorGraphs[-1].var_value)
            print(ufo_to_master)


            if tag == messages.INFER:
                data = {"pid": partition_id,
                        "values": messages.serialize(variables_to_master),
                        "pf": messages.serialize(pf_to_master),
                        "ufo": messages.serialize(ufo_to_master)}
                __salt__['event.send'](messages.INFER_RES, data)
            else:
                messages.compute_vars_to_send(map_to_master, var_evid_to_master, ns_minion.ns.factorGraphs[-1].var_value_evid[0])
                messages.compute_pf_values(factor, fmap, ns_minion.ns.factorGraphs[-1].var_value_evid, variable, pf_list, pf_evid_to_master)
                messages.compute_ufo_values(factor, fmap, ns_minion.ns.factorGraphs[-1].var_value_evid, variable, var_ufo, ufo_send, ufo_start, ufo_map, ufo_evid_to_master)
                dweight = ns_minion.ns.factorGraphs[-1].weight_value[0] - w0

                data = {"pid": partition_id,
                        "values": messages.serialize(variables_to_master),
                        "v_evid": messages.serialize(var_evid_to_master),
                        "pf": messages.serialize(pf_to_master),
                        "pf_evid": messages.serialize(pf_evid_to_master),
                        "ufo": messages.serialize(ufo_to_master),
                        "ufo_evid": messages.serialize(ufo_to_master),
                        "dw": messages.serialize(dweight)}
                __salt__['event.send'](messages.LEARN_RES, data)
        loop_end = time.time()
        print("*****" + tag + " took " + str(loop_end - loop_begin) + "*****")
