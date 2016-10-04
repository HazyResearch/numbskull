#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import python libs
from __future__ import absolute_import
import json
import logging
import os.path
import numbskull
from numbskull import numbskull
import argparse
import sys
import numpy as np
import codecs

# Import salt libs
import salt.utils.event
import salt.client
import salt.runner
import salt.config
import nb_syspaths as syspaths

import messages
import time


class NumbskullMaster:
    def __init__(self, argv):
        # Salt conf init
        self.master_conf_dir = os.path.join(syspaths.CONFIG_DIR, 'master')
        self.salt_opts = salt.config.client_config(self.master_conf_dir)
        # Salt cliens init
        self.local_client = salt.client.LocalClient(self.master_conf_dir)
        self.runner = salt.runner.RunnerClient(self.salt_opts)
        # Salt event bus - used to communicate with minions
        self.event_bus = salt.utils.event.get_event(
                             'master',
                             sock_dir=self.salt_opts['sock_dir'],
                             transport=self.salt_opts['transport'],
                             opts=self.salt_opts)
        # Get active minions
        #self.minions = self.get_minions_status()
        # Numbskull-related variables
        self.argv = argv
        self.args = self.parse_args(argv)
        self.ns = None
        self.num_minions = 2

    def initialize(self):
        self.assign_partition_id()
        self.prepare_db()
        self.prep_numbskull()
        self.load_own_fg()
        self.load_minions_fg()

    def inference(self):
        # TODO: switch to proper probs
        marginals = {}
        for fgID in range(len(self.ns.factorGraphs)):
            marginals[fgID] = []
            # Run learning on minions
            SUCCESS, m_marginals = self.inference_minions(fgID)
            if not SUCCESS:
                print 'Minions-learning failed'
                return
            marginals[fgID].extend(m_marginals)
            # Run learning locally
            self.ns.inference(fgID, False)
            marginals[fgID].append(self.ns.factorGraphs[fgID].marginals)
            # Combine results
        return marginals

    def learning(self):
        weights = {}
        for fgID in range(len(self.ns.factorGraphs)):
            weights[fgID] = []
            # Run learning on minions
            SUCCESS, m_weights = self.learning_minions(fgID)
            if not SUCCESS:
                print 'Minions-learning failed'
                return
            weights[fgID].extend(m_weights)
            # Run learning locally
            self.ns.learning(fgID, False)
            weights[fgID].append(self.ns.factorGraphs[fgID].weight_value)
            # Combine results
        return weights

    # Init Phase
    def assign_partition_id(self):
        while True:
            self.minions = self.get_minions_status()['up']
            if len(self.minions) >= self.num_minions:
                break
            print("Waiting for minions")
            time.sleep(1)
        print("Minions obtained")

        self.minions = self.minions[:self.num_minions]
        for (i, name) in enumerate(self.minions):
            data = {'id': i}
            newEvent = self.local_client.cmd([name],
                                             'event.fire',
                                             [data, messages.ASSIGN_ID],
                                             expr_form='list')
        # TODO: listen for responses

    def prepare_db(self):
        # TODO: implement
        pass

    def prep_numbskull(self):
        # Setup local instance
        self.prep_local_numbskull()
        # Setup minion instnances
        success = self.prep_minions_numbskull()
        if not success:
            print 'ERROR: Numbksull not loaded'

    def load_own_fg(self):
        # TODO: implement
        # Needs to load the factor graph
        # Track what to sample
        # Track map for variables/factors from each minion
        pass

    def load_minions_fg(self):
        for fg in self.ns.factorGraphs:
            if not self.send_minions_fg_data(fg):
                print 'ERROR: Could not send FG to minions'
                return
        print 'SUCCESS: FG loaded to all minions'
        return

    # Inference
    def inference_minions(self, fgID):
        # Prep event
        tag = messages.INFER
        data = {'fgID': fgID}
        newEvent = self.local_client.cmd(self.minions,
                                         'event.fire',
                                         [data, tag],
                                         expr_form='list')
        # variable props for data from minions
        SUCCESS = True
        resp = []
        marginals = []
        while len(resp) < len(self.minions):
            evdata = self.event_bus.get_event(wait=5,
                                              tag=messages.INFER_RES,
                                              full=True)
            if evdata:
                tag, data = evdata['tag'], evdata['data']
                jevent = json.dumps(data)
                if data['data']['status'] != 'SUCCESS':
                    print 'ERROR: Minion %s failed to run inference.' % data['id']
                    SUCCESS = False
                else:
                    m = self.deserialize(data['data']['marginals'], np.float64)
                    marginals.append(m)
            resp.append((data['id'], SUCCESS))
        if SUCCESS:
            print 'SUCCESS: All minions ran inference.'
        return SUCCESS, marginals



    # Helper
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

    def get_minions_status(self):
        minion_status = self.runner.cmd('manage.status')
        print "***** MINION STATUS REPORT *****"
        print minion_status
        print "UP: ", len(minion_status['up'])
        print "DOWN: ", len(minion_status['down'])
        print
        return minion_status

    def prep_local_numbskull(self):
        self.ns = numbskull.NumbSkull(**vars(self.args))
        #self.ns.loadFGFromFile()
        #fg = self.ns.factorGraphs[-1]
        #print 'Weights: %d' % fg.weight.shape[0]
        #print 'Variables: %d' % fg.variable.shape[0]
        #print 'Factors:  %d' % fg.factor.shape[0]

    def prep_minions_numbskull(self):
        # send args and initialize numbskull at minion
        data = {'argv': self.argv}
        newEvent = self.local_client.cmd(self.minions,
                                         'event.fire',
                                         [data, messages.INIT_NS],
                                         expr_form='list')
        # wait for ACK from minions
        SUCCESS = True
        resp = []
        while len(resp) < len(self.minions):
            evdata = self.event_bus.get_event(wait=5,
                                              tag=messages.INIT_NS_RES,
                                              full=True)
            if evdata:
                tag, data = evdata['tag'], evdata['data']
                jevent = json.dumps(data)
                if data['data']['status'] == 'FAIL':
                    print 'ERROR: Minion %s failed to load numbskull.' % data['id']
                    SUCCESS = False
                resp.append((data['id'], SUCCESS))
        if SUCCESS:
            print 'SUCCESS: All minions loaded numbskull.'
        return SUCCESS

    def serialize(self, array):
        return array.tostring().decode('utf16').encode('utf8')

    def deserialize(self, array, dtype):
        ar = array.decode('utf8').encode('utf16').lstrip(codecs.BOM_UTF16)
        return np.fromstring(ar, dtype)

    def minion_fg_data(self, fg):
        data = {}
        data['weight'] = fg.weight.tostring()
        data['variable'] = fg.variable.tostring()
        data['factor'] = self.serialize(fg.factor)
        data['fmap'] = fg.fmap.tostring()
        data['vmap'] = fg.vmap.tostring()
        data['factor_index'] = fg.factor_index.tostring()
        return data

    def send_minions_fg_data(self, fg):
        # Prep event
        tag = messages.LOAD_FG
        data = self.minion_fg_data(fg)
        newEvent = self.local_client.cmd(self.minions,
                                         'event.fire',
                                         [data, tag],
                                         expr_form='list')
        # wait for ACK from minions
        SUCCESS = True
        resp = []
        while len(resp) < len(self.minions):
            evdata = self.event_bus.get_event(wait=5,
                                              tag=messages.LOAD_FG_RES,
                                              full=True)
            if evdata:
                tag, data = evdata['tag'], evdata['data']
                jevent = json.dumps(data)
                if data['data']['status'] != 'SUCCESS':
                    print 'ERROR: Minion %s failed to load FG.' % data['id']
                    SUCCESS = False
                else:
                    print 'SUCCESS: Minion %s factor graph stats:' % data['id']
                    print 'Variables: %d' % data['data']['meta']['variables']
                    print 'Weights: %d' % data['data']['meta']['weights']
                    print 'Factors: %d' % data['data']['meta']['factors']
                resp.append((data['id'], SUCCESS))
        if SUCCESS:
            print 'SUCCESS: All minions loaded FG.'
        return SUCCESS

    def learning_minions(self, fgID):
        # Prep event
        tag = messages.LEARN
        data = {'fgID': fgID}
        newEvent = self.local_client.cmd(self.minions,
                                         'event.fire',
                                         [data, tag],
                                         expr_form='list')
        # weight for data from minions
        SUCCESS = True
        resp = []
        weights = []
        while len(resp) < len(self.minions):
            evdata = self.event_bus.get_event(wait=5,
                                              tag=messages.LEARN_RES,
                                              full=True)
            if evdata:
                tag, data = evdata['tag'], evdata['data']
                jevent = json.dumps(data)
                if data['data']['status'] != 'SUCCESS':
                    print 'ERROR: Minion %s failed to run learning.' % data['id']
                    SUCCESS = False
                else:
                    w = self.deserialize(data['data']['weights'], np.float64)
                    weights.append(w)
                resp.append((data['id'], SUCCESS))
        if SUCCESS:
            print 'SUCCESS: All minions ran learning.'
        return SUCCESS, weights


def main(argv=None):
    args = ['../../test',
            '-l', '100',
            '-i', '100',
            '-t', '10',
            '-s', '0.01',
            '--regularization', '2',
            '-r', '0.1',
            '--quiet']

    ns_master = NumbskullMaster(args)
    ns_master.initialize()
    #w = ns_master.learning()
    p = ns_master.inference()
    return ns_master, w, p

if __name__ == "__main__":
    main()
