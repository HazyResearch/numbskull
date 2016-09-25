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
try:
    pass
    import numbskull
    from numbskull import numbskull
    from numbskull.numbskulltypes import *
except:
    print 'ERROR: Cannot load numbskull'
    sys.exit(-1)

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
    while True:
        evdata = event_bus.get_event(full=True)
        if evdata:
            tag, data = evdata['tag'], evdata['data']
            jevent = json.dumps(data)
            log.debug(jevent)
            if data:
                if tag == 'INIT_NS':
                    try:
                        ns_minion.init_numbskull(data['argv'])
                        # Respond OK to master
                        tag = 'INIT_NS_RES'
                        data = {'status': 'OK'}
                        __salt__['event.send'](tag, data)
                    except:
                        # Respond FAIL to master
                        tag = 'INIT_NS_RES'
                        data = {'status': 'FAIL'}
                        __salt__['event.send'](tag, data)
                elif tag == 'LOAD_FG':
                    status, meta = ns_minion.loadFG(data)
                    # Respond to master
                    tag = 'LOAD_FG_RES'
                    data = {'status': status, 'meta': meta}
                    __salt__['event.send'](tag, data)
                elif tag == 'LEARN':
                    status, weights = ns_minion.learning(data['fgID'])
                    # Respond to master
                    tag = 'LEARN_RES'
                    data = {'status': status, 'weights': weights}
                    __salt__['event.send'](tag, data)
                elif tag == 'INFER':
                    status, marginals = ns_minion.inference(data['fgID'])
                    # Respond to master
                    tag = 'INFER_RES'
                    data = {'status': status, 'marginals': marginals}
                    __salt__['event.send'](tag, data)
