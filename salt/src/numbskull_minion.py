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

# Import salt libs
import salt.utils.event

# Import numbskull
m_opts = salt.config.minion_config(os.environ['SALT_CONFIG_DIR']+'/minion')
sys.path.append(m_opts['extension_modules']+'/modules')
import numbskull
from numbskull import numbskull
from numbskull.numbskulltypes import *

import messages
import pydoc
import psycopg2
import urlparse
import numpy as np


log = logging.getLogger(__name__)


class NumbskullMinion:
    def __init__(self):
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
        self.args = self.parse_args(argv)
        self.ns = numbskull.NumbSkull(**vars(self.args))

    def serialize(self, array):
        return array.tobytes().decode('utf16').encode('utf8')

    def deserialize(self, array, dtype):
        ar = array.decode('utf8').encode('utf16').lstrip(codecs.BOM_UTF16)
        return np.fromstring(ar, dtype=dtype)

    def loadFG(self, data):
        try:
            weight = np.fromstring(data['weight'], dtype=Weight)
            variable = np.fromstring(data['variable'], dtype=Variable)
            factor = self.deserialize(data['factor'], Factor)
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
        try:
            self.ns.learning(fgID, False)
            weights = self.ns.factorGraphs[fgID].weight_value
            return 'SUCCESS', self.serialize(weights)
        except:
            return 'FAILED', None

    def inference(self, fgID):
        try:
            self.ns.inference(fgID, False)
            marginals = self.ns.factorGraphs[fgID].marginals
            return 'SUCCESS', self.serialize(marginals)
        except:
            return 'FAILED', None


def start():
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
    while True:
        evdata = event_bus.get_event(full=True)
        log.debug('DEBUG 1')
        if evdata:
            log.debug('DEBUG 2')
            tag, data = evdata['tag'], evdata['data']
            jevent = json.dumps(data)
            log.debug(jevent)
            if data:
                log.debug("DEBUG 3")
                #strhelp = pydoc.render_doc(messages, "Help on %s")
                #log.debug(strhelp)
                log.debug(tag)
                #log.debug(messages.INIT_NS)
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
                    # TODO: actually should be loading from database
                    # Needs to compute 
                    # Track what to sample
                    # Track map for variables/factors from each minion

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
                        database = database,
                        user = username,
                        password = password,
                        host = hostname,
                        port = port
                    )

                    # Open a cursor to perform database operations
                    cur = conn.cursor()
                    (factor_view, variable_view, weight_view) = messages.get_views(cur)
                    minion_filter = "   partition_key = 'B' " \
                                    "or partition_key = 'C{partition_id}' " \
                                    "or partition_key = 'D{partition_id}' " \
                                    "or partition_key = 'E{partition_id}' " \
                                    "or partition_key = 'F{partition_id}' " \
                                    "or partition_key = 'G{partition_id}' " \
                                    "or partition_key = 'H' "
                    minion_filter = minion_filter.format(partition_id=partition_id)

                    # TODO: factors
                    var_data = messages.read_views(cur, variable_view, minion_filter)

                    # Load variable info
                    var_data = messages.read_views(cur, variable_view, minion_filter)
                    vid = np.zeros(len(var_data), np.int64)
                    variable = np.zeros(len(var_data), Variable)
                    var_pt = np.zeros(len(var_data), np.str_) # partition type
                    for (i, v) in enumerate(var_data):
                        vid[i] = v[0]
                        variable[i]["isEvidence"] = v[1]
                        variable[i]["initialValue"] = v[2]
                        variable[i]["dataType"] = v[3]
                        variable[i]["cardinality"] = v[4]
                        #variable[i]["vtf_offset"] = ???
                        var_pt[i] = v[5][0] # only first char needed now
                                            # (partition id will match)

                    perm = vid.argsort()
                    vid = vid[perm]
                    variable = variable[perm]
                    var_pt = var_pt[perm]
                    # TODO: vtf_offset

                    # TODO: weights

                    # Close communication with the database
                    cur.close()
                    conn.close()

                    log.debug("DONE LOAD_FG")
                    status, meta = ns_minion.loadFG(data)
                    # Respond to master
                    data = {'status': status, 'meta': meta}
                    __salt__['event.send'](messages.LOAD_FG_RES, data)
                elif tag == messages.LEARN:
                    status, weights = ns_minion.learning(data['fgID'])
                    # Respond to master
                    data = {'status': status, 'weights': weights}
                    __salt__['event.send'](messages.LEARN_RES, data)
                elif tag == messages.INFER:
                    status, marginals = ns_minion.inference(data['fgID'])
                    # Respond to master
                    data = {'status': status, 'marginals': marginals}
                    __salt__['event.send'](messages.INFER_RES, data)
