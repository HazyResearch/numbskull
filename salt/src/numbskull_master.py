#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import python libs
from __future__ import print_function, absolute_import
import json
import logging
import os.path
import numbskull
from numbskull import numbskull
import argparse
import sys
import subprocess
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

import psycopg2
import urlparse

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
        self.prep_numbskull()
        db_url = self.prepare_db()
        self.load_own_fg()
        self.load_minions_fg(db_url)
        self.sync_mapping()

    def inference(self, epochs=1):
        # TODO: switch to proper probs

        print("BEGINNING INFERENCE")
        begin = time.time()
        variables_to_minions = np.zeros(self.map_to_minions.size, np.int64)
        for i in range(epochs):
            print("Inference loop", i)
            # sample own variables
            begin1 = time.time()
            fgID = 0
            # TODO: do not sample vars owned by minion
            self.ns.inference(fgID, False)
            end1 = time.time()
            print("INFERENCE LOOP TOOK " + str(end1 - begin1))


            # gather values to ship to minions
            # TODO: handle multiple copies
            for i in range(self.map_to_minions.size):
                variables_to_minions[i] = self.ns.factorGraphs[-1].var_value[0][self.map_to_minions[i]]
            #print(variables_to_minions)
            #print(self.ns.factorGraphs[-1].var_value)

            # Tell minions to sample
            tag = messages.INFER
            data = {"values": messages.serialize(variables_to_minions)}
            newEvent = self.local_client.cmd(self.minions,
                                             'event.fire',
                                             [data, tag],
                                             expr_form='list')

            # TODO: receive values from minions
            resp = 0
            while resp < len(self.minions):
                evdata = self.event_bus.get_event(wait=5,
                                                  tag=messages.INFER_RES,
                                                  full=True)
                if evdata:
                    resp += 1
                    data = evdata['data']['data']
                    #print(data)
                    pid = data["pid"]
                    variables_from_minion = messages.deserialize(data["values"], np.int64)
                    for i in range(variables_from_minion.size):
                        self.ns.factorGraphs[-1].var_value[0][self.map_from_minion[pid][i]] = variables_from_minion[i]

                #if evdata:
                #    tag, data = evdata['tag'], evdata['data']
                #    jevent = json.dumps(data)
                #    if data['data']['status'] != 'SUCCESS':
                #        print('ERROR: Minion %s failed to run inference.' % data['id'])
                #        SUCCESS = False
                #    else:
                #        #m = self.deserialize(data['data']['marginals'], np.float64)
                #        #marginals.append(m)
                #        pass
                #resp.append((data['id'], SUCCESS))
                #resp.append((data['id'], SUCCESS))

            # TODO: get marginals
        end = time.time()
        print("INFERENCE TOOK", end - begin)

        return
        marginals = {}
        for fgID in range(len(self.ns.factorGraphs)):
            marginals[fgID] = []
            # Run learning on minions
            SUCCESS, m_marginals = self.inference_minions(fgID)
            if not SUCCESS:
                print('Minions-learning failed')
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
                print('Minions-learning failed')
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

        # hard-coded application directory
        #application_dir = "/dfs/scratch0/bryanhe/genomics/"
        #application_dir = "/dfs/scratch0/bryanhe/census/"
        #application_dir = "/dfs/scratch0/bryanhe/voting/"
        application_dir = "/dfs/scratch0/bryanhe/congress2/"

        # obtain database url from file
        with open(application_dir + "/db.url", "r") as f:
            db_url = f.read().strip()

        # Call deepdive to perform everything up to grounding
        # TODO: is there a way to not "do all" (which even uses DW sampling)
        #       and just ground?
        # TODO: check that deepdive ran successfully
        subprocess.call(["deepdive", "do", "process/grounding/combine_factorgraph"], cwd=application_dir)

        # Obtain partition information
        # TODO: remove hard-coded 2
        partition_json = subprocess.check_output(["ddlog", "semantic-partition", "app.ddlog", "--ppa", "-w", "2"], cwd=application_dir)
        partition = json.loads(partition_json)

        # Connect to an existing database
        # http://stackoverflow.com/questions/15634092/connect-to-an-uri-in-postgres
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

        # Select which partitioning
        # TODO: actually check costs
        print(len(partition))
        print(partition[0].keys())
        print("********************************************************************************")
        for p in partition:

            print(p["partition_types"])
            cur.execute(p["sql_to_cost"])
            cost = cur.fetchone()[0]
            print(cost)
            if p["partition_types"] == "(1)":
            #if p["partition_types"] == "":
            #if p["partition_types"] == "(0)":
                p0 = p
        print("********************************************************************************")

        # p0 is partition to use
        for k in p0.keys():
            print(k)
            print(p0[k])
            print()


        # This adds partition information to the database
        for op in p0["sql_to_apply"]:
            #print(op)
            # Currently ignoring the column already exists from ALTER statements
            # TODO: better fix?
            try:
                cur.execute(op)
                # Make the changes to the database persistent
                conn.commit()
            except psycopg2.ProgrammingError:
                print("Unexpected error:", sys.exc_info())
                conn.rollback()

        (factor_view, variable_view, weight_view) = messages.get_views(cur)
        print(factor_view)
        print(variable_view)
        print(weight_view)
        master_filter = "   partition_key = 'A' " \
                        "or partition_key = 'B' " \
                        "or partition_key like 'D%' " \
                        "or partition_key like 'F%' " \
                        "or partition_key like 'G%' " \
                        "or partition_key like 'H%' "


        (weight, variable, factor, fmap, domain_mask, edges, self.var_pt, self.var_pid, self.factor_pt, self.factor_pid, self.vid) = messages.get_fg_data(cur, master_filter)
        print(self.vid)
        print(self.var_pt)
        print(self.var_pid)
        print(self.factor_pt)
        print(self.factor_pid)

        # Close communication with the database
        cur.close()
        conn.close()

        self.ns.loadFactorGraph(weight, variable, factor, fmap, domain_mask, edges)

        # Close communication with the database
        cur.close()
        conn.close()
        return db_url


    def prep_numbskull(self):
        # Setup local instance
        self.prep_local_numbskull()
        # Setup minion instnances
        success = self.prep_minions_numbskull()
        if not success:
            print('ERROR: Numbksull not loaded')

    def load_own_fg(self):
        # TODO: implement
        # Needs to load the factor graph
        # Track what to sample
        # Track map for variables/factors from each minion
        pass

    def load_minions_fg(self, db_url):
        tag = messages.LOAD_FG
        data = {"db_url": db_url}
        newEvent = self.local_client.cmd(self.minions,
                                         'event.fire',
                                         [data, tag],
                                         expr_form='list')

        print("WAITING FOR MINION LOAD_FG_RES")
        resp = 0
        while resp < len(self.minions):
            evdata = self.event_bus.get_event(wait=5,
                                              tag=messages.LOAD_FG_RES,
                                              full=True)
            if evdata:
                resp += 1
        print("DONE WAITING FOR MINION LOAD_FG_RES")
        #for fg in self.ns.factorGraphs:
        #    if not self.send_minions_fg_data(fg):
        #        print('ERROR: Could not send FG to minions')
        #        return
        #print('SUCCESS: FG loaded to all minions')
        #return

    def sync_mapping(self):
        # compute map
        l = 0
        for i in range(len(self.var_pt)):
            if self.var_pt[i] == "B":
                l += 1

        self.map_to_minions = np.zeros(l, np.int64)
        l = 0
        for i in range(len(self.var_pt)):
            if self.var_pt[i] == "B":
                self.map_to_minions[l] = self.vid[i]
                l += 1
        print(self.map_to_minions)

        # send mapping to minions
        tag = messages.SYNC_MAPPING
        data = {"map": messages.serialize(self.map_to_minions)}
        newEvent = self.local_client.cmd(self.minions,
                                         'event.fire',
                                         [data, tag],
                                         expr_form='list')

        self.map_from_minion = [None for i in range(len(self.minions))]
        resp = 0
        while resp < len(self.minions):
            # TODO: receive map and save
            evdata = self.event_bus.get_event(wait=5,
                                              tag=messages.SYNC_MAPPING_RES,
                                              full=True)
            if evdata:
                print(evdata)
                tag, data = evdata['tag'], evdata['data']['data']
                print(data)
                self.map_from_minion[data["pid"]] = messages.deserialize(data["map"], np.int64)
                resp += 1
        print("DONE WITH SENDING MAPPING")

        for i in range(len(self.map_to_minions)):
            self.map_to_minions[i] = messages.inverse_map(self.vid, self.map_to_minions[i])

        for i in range(len(self.map_from_minion)):
            for j in range(len(self.map_from_minion[i])):
                self.map_from_minion[i][j] = messages.inverse_map(self.vid, self.map_from_minion[i][j])


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
                    print('ERROR: Minion %s failed to run inference.' % data['id'])
                    SUCCESS = False
                else:
                    m = messages.deserialize(data['data']['marginals'], np.float64)
                    marginals.append(m)
            resp.append((data['id'], SUCCESS))
        if SUCCESS:
            print('SUCCESS: All minions ran inference.')
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
        print("***** MINION STATUS REPORT *****")
        print(minion_status)
        print("UP: ", len(minion_status['up']))
        print("DOWN: ", len(minion_status['down']))
        print()
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
                    print('ERROR: Minion %s failed to load numbskull.' % data['id'])
                    SUCCESS = False
                resp.append((data['id'], SUCCESS))
        if SUCCESS:
            print('SUCCESS: All minions loaded numbskull.')
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
                    print('ERROR: Minion %s failed to load FG.' % data['id'])
                    SUCCESS = False
                else:
                    print('SUCCESS: Minion %s factor graph stats:' % data['id'])
                    print('Variables: %d' % data['data']['meta']['variables'])
                    print('Weights: %d' % data['data']['meta']['weights'])
                    print('Factors: %d' % data['data']['meta']['factors'])
                resp.append((data['id'], SUCCESS))
        if SUCCESS:
            print('SUCCESS: All minions loaded FG.')
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
                    print('ERROR: Minion %s failed to run learning.' % data['id'])
                    SUCCESS = False
                else:
                    w = self.deserialize(data['data']['weights'], np.float64)
                    weights.append(w)
                resp.append((data['id'], SUCCESS))
        if SUCCESS:
            print('SUCCESS: All minions ran learning.')
        return SUCCESS, weights


def main(argv=None):
    args = ['../../test',
            '-l', '1',
            '-i', '1',
            '-t', '1',
            '-s', '0.01',
            '--regularization', '2',
            '-r', '0.1',
            '--quiet']

    ns_master = NumbskullMaster(args)
    ns_master.initialize()
    #w = ns_master.learning()
    p = ns_master.inference(100)
    #p = ns_master.inference(100)
    #return ns_master, w, p
    return ns_master

if __name__ == "__main__":
    main()
