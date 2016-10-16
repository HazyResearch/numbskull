"""TODO."""

# Import python libs
from __future__ import absolute_import
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
            minion_filter = "   partition_key = 'B' " \
                            "or partition_key = 'C{partition_id}' " \
                            "or partition_key = 'D{partition_id}' " \
                            "or partition_key = 'E{partition_id}' " \
                            "or partition_key = 'F{partition_id}' " \
                            "or partition_key = 'G{partition_id}' " \
                            "or partition_key = 'H' "
            minion_filter = minion_filter.format(partition_id=partition_id)

            (weight, variable, factor, fmap, domain_mask, edges,
                var_pt, var_pid, factor_pt, factor_pid, vid) = \
                messages.get_fg_data(cur, minion_filter)

            # Close communication with the database
            cur.close()
            conn.close()

            for (i, v) in enumerate(variable):
                # B is only variable partition type on minions but not owned
                if var_pt[i] == "B":
                    v["isEvidence"] = 4 # not owned var type

            ns_minion.ns.loadFactorGraph(weight, variable, factor, fmap,
                                         domain_mask, edges)

            # Respond to master
            data = {}
            __salt__['event.send'](messages.LOAD_FG_RES, data)
            log.debug("DONE LOADFG")
        elif tag == messages.SYNC_MAPPING:
            # receive map from master
            map_from_master = messages.deserialize(data["map"], np.int64)
            #log.debug(map_from_master)

            # compute map
            l = 0
            for i in range(len(var_pt)):
                if var_pt[i] == "D":
                    l += 1

            map_to_master = np.zeros(l, np.int64)
            l = 0
            for i in range(len(var_pt)):
                if var_pt[i] == "D":
                    map_to_master[l] = vid[i]
                    l += 1
            #log.debug(map_to_master)

            for i in range(len(map_from_master)):
                map_from_master[i] = \
                        messages.inverse_map(vid, map_from_master[i])

            for i in range(len(map_to_master)):
                map_to_master[i] = \
                        messages.inverse_map(vid, map_to_master[i])
            variables_to_master = np.zeros(map_to_master.size, np.int64)
            
            data = {"pid": partition_id,
                    "map": messages.serialize(map_to_master)}
            __salt__['event.send'](messages.SYNC_MAPPING_RES, data)
            log.debug("DONE SYNC_MAPPING")
        elif tag == messages.LEARN:
            status, weights = ns_minion.learning(data['fgID'])
            # Respond to master
            data = {'status': status, 'weights': weights}
            __salt__['event.send'](messages.LEARN_RES, data)
        elif tag == messages.INFER:
            variables_from_master = \
                messages.deserialize(data["values"], np.int64)
            for i in range(map_from_master.size):
                m = map_from_master[i]
                v = variables_from_master[i]
                ns_minion.ns.factorGraphs[-1].var_value[0][m] = v

            begin = time.time()
            # TODO: do not sample variables owned by master
            status, marginals = ns_minion.inference()
            end = time.time()
            log.debug("INFERENCE LOOP TOOK " + str(end - begin))

            # Respond to master
            for (i, m) in enumerate(map_to_master):
                variables_to_master[i] = \
                    ns_minion.ns.factorGraphs[-1].var_value[0][m]

            data = {"pid": partition_id,
                    "values": messages.serialize(variables_to_master)}
            __salt__['event.send'](messages.INFER_RES, data)
        loop_end = time.time()
        print("**********" + tag + " took " + str(loop_end - loop_begin) + "**********")
