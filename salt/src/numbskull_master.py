#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TODO."""

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
from multiprocessing import Pool
from functools import partial

# Import salt libs
import salt.utils.event
import salt.client
import salt.runner
import salt.config

import messages
import time

import psycopg2
import urlparse
import numbskull_master_client
from numbskull_master_client import InfLearn_Channel

master_conf_dir = \
            os.path.join(os.environ['SALT_CONFIG_DIR'], 'master')
salt_opts = salt.config.client_config(master_conf_dir)


def send_to_minion(data, tag, tgt):
    """TODO."""
    salt_opts['minion_uri'] = 'tcp://{ip}:{port}'.format(
        ip=salt.utils.ip_bracket(tgt),
        port=7341  # TODO, no fallback
    )
    load = {'id': 'master_inflearn',
            'tag': tag,
            'data': data}
    channel = InfLearn_Channel.factory(salt_opts)
    channel.send(load)
    return True


class NumbskullMaster:
    """TODO."""

    def __init__(self, application_dir, machines, 
                 partition_method, partition_type, argv):
        """TODO."""
        # Salt conf init
        self.master_conf_dir = master_conf_dir
        self.salt_opts = salt_opts

        # Salt clients init
        self.local_client = salt.client.LocalClient(self.master_conf_dir)
        self.runner = salt.runner.RunnerClient(self.salt_opts)

        # Salt event bus - used to communicate with minions
        self.event_bus = salt.utils.event.get_event(
                             'master',
                             sock_dir=self.salt_opts['sock_dir'],
                             transport=self.salt_opts['transport'],
                             opts=self.salt_opts)

        # Numbskull-related variables
        self.argv = argv
        self.args = self.parse_args(argv)
        self.ns = None
        self.application_dir = application_dir
        self.num_minions = machines
        self.partition_type = partition_type

        # Partitioning variables
        self.part_method = partition_method

        # DB variables
        self.db_url = None
        self.conn = None

    def initialize(self):
        """TODO."""
        time1 = time.time()
        self.assign_partition_id()
        time2 = time.time()
        print("assign_partition_id took " + str(time2 - time1))

        self.prep_numbskull()
        time3 = time.time()
        print("prep_numbskull took " + str(time3 - time2))

        out = self.prepare_db()
        time4 = time.time()
        print("prepare_db took " + str(time4 - time3))

        self.load_own_fg()
        time5 = time.time()
        print("load_own_fg took " + str(time5 - time4))

        self.load_minions_fg(db_url)
        time6 = time.time()
        print("load_minions_fg took " + str(time6 - time5))

        self.sync_mapping()
        time7 = time.time()
        print("sync_mapping took " + str(time7 - time6))

    # The code for learning and inference share a lot of code (computing
    # variable map, handling partial factors) so they are in one func).
    # This is just a trivial wrapper function.
    def learning(self, epochs=1):
        """TODO."""
        return self.inference(epochs, True)

    def inference(self, epochs=1, learn=False):
        """TODO."""
        mode = "LEARNING" if learn else "INFERENCE"
        print("BEGINNING " + mode)
        begin = time.time()
        variables_to_minions = np.zeros(self.map_to_minions.size, np.int64)

        for i in range(epochs):
            print("Inference loop", i)
            # sample own variables
            begin1 = time.time()
            fgID = 0

            if learn:
                self.ns.learning(fgID, False)
            else:
                self.ns.inference(fgID, False)

            end1 = time.time()
            print(mode + " LOOP TOOK " + str(end1 - begin1))

            # gather values to ship to minions
            # TODO: handle multiple copies
            for (i, m) in enumerate(self.map_to_minions):
                variables_to_minions[i] = \
                        self.ns.factorGraphs[-1].var_value[0][m]

            # Tell minions to sample
            if learn:
                tag = messages.LEARN
                beginTest = time.time()
                # TODO: which copy of weight to use when multiple
                weight_value = self.ns.factorGraphs[-1].weight_value[0]
                data = {"values": messages.serialize(variables_to_minions),
                        "weight": messages.serialize(weight_value)}
            else:
                tag = messages.INFER
                beginTest = time.time()
                data = {"values": messages.serialize(variables_to_minions)}

            if self.num_minions != 0:
                pub_func = partial(send_to_minion, data, tag)
                self.clientPool.imap(pub_func, self.minion2host.values())

            endTest = time.time()
            print("EVENT FIRE LOOP TOOK " + str(endTest - beginTest))

            resp = 0
            while resp < len(self.minions):
                tag = messages.LEARN_RES if learn else messages.INFER_RES
                evdata = self.event_bus.get_event(wait=5,
                                                  tag=tag,
                                                  full=True)
                if evdata:
                    resp += 1
                    data = evdata['data']['data']
                    pid = data["pid"]
                    # Process variables from minions
                    vfmin = messages.deserialize(data["values"], np.int64)
                    for (i, v) in enumerate(vfmin):
                        m = self.map_from_minion[pid][i]
                        self.ns.factorGraphs[-1].var_value[0][m] = v

                    if learn:
                        self.ns.factorGraphs[-1].weight_value[0] += \
                            messages.deserialize(data["dw"], np.float64)

        # TODO: get and return marginals
        # TODO: switch to proper probs
        end = time.time()
        print(mode + " TOOK", end - begin)
        return end - begin

    ##############
    # Init Phase #
    ##############
    def assign_partition_id(self):
        """TODO."""
        while True:
            self.minions = self.get_minions_status()['up']
            if len(self.minions) >= self.num_minions:
                break
            print("Waiting for minions (" + str(len(self.minions)) +
                  " / " + str(self.num_minions) + ")")
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
        # Obtain minions ip addresses
        self.minion2host = \
            self.local_client.cmd(self.minions, 'grains.get', ['localhost'],
                                  expr_form='list', timeout=None)
        # Initialize multiprocessing pool for publishing
        if self.num_minions != 0:
            self.clientPool = Pool(self.num_minions)

    def prep_numbskull(self):
        """TODO."""
        # Setup local instance
        self.prep_local_numbskull()
        # Setup minion instances
        success = self.prep_minions_numbskull()
        if not success:
            print('ERROR: Numbksull not loaded')

    def open_db_connection(self):
        # obtain database url from file
        with open(self.application_dir + "/db.url", "r") as f:
            self.db_url = f.read().strip()

        # Connect to an existing database
        # http://stackoverflow.com/questions/15634092/connect-to-an-uri-in-postgres
        url = urlparse.urlparse(self.db_url)
        username = url.username
        password = url.password
        database = url.path[1:]
        hostname = url.hostname
        port = url.port
        self.conn = psycopg2.connect(
            database=database,
            user=username,
            password=password,
            host=hostname,
            port=port
        )

    def run_deepdive(self):
        # Call deepdive to perform everything up to grounding
        # TODO: check that deepdive ran successfully
        cmd = ["deepdive", "do", "all"]
        subprocess.call(cmd, cwd=self.application_dir)

    def run_ddlog(self):
        # semantic partitioning
        if self.partition_method == 'sp':
            cmd = ["ddlog", "semantic-partition", "app.ddlog",
               "--ppa",
               # "-u",
               "--workers", str(self.num_minions),
               "--cost-model", "simple.costmodel.txt"]
            partition_json = subprocess.check_output(cmd, 
                                         cwd=self.application_dir)
            partition = json.loads(partition_json)
            return partition
        # Metis or connected components based partitioning
        elif self.partition_method == 'metis' or self.partition_method == 'cc':
            cmd = ["ddlog", "cc-partition", "app.ddlog",
               "--workers", str(self.num_minions)]
            partition_json = subprocess.check_output(cmd,
                                         cwd=self.application_dir)
            partition = json.loads(partition_json)
            return partition
        # Default
        else:
            print('ERROR: Invalid partition method!')
            return False

    def get_fg(self, cur):
        """TODO"""
        master_filter = "   partition_key = 'A' " \
                        "or partition_key = 'B' " \
                        "or partition_key like 'D%' " \
                        "or partition_key like 'F%' " \
                        "or partition_key like 'G%' " \
                        "or partition_key like 'H%' "
        get_fg_data_begin = time.time()
        (weight, variable, factor, fmap, domain_mask, edges, self.var_pt,
         self.factor_pt, self.vid) = messages.get_fg_data(cur, master_filter)
        get_fg_data_end = time.time()
        print("Done running get_fg_data: " +
              str(get_fg_data_end - get_fg_data_begin))

        # TODO: could be in numba
        for (i, v) in enumerate(variable):
            # D is only variable partition type on master but not owned
            if self.var_pt[i] == "D":
                v["isEvidence"] = 4  # not owned var type

        self.ns.loadFactorGraph(weight, variable, factor, fmap,
                                domain_mask, edges)

    def prepare_db(self):
        """TODO."""
        # Run deepdive to perform candidate extraction
        self.run_deepdive()

        # Obtain partition information
        partition = self.run_ddlog()
        if not partition:
            return False

        # Open DB connection
        self.open_db_connection()

        # Check if partioning is metis or cc
        if self.partition_method == 'metis':
            messages.find_metis_parts(self.conn, self.num_minions)
        elif self.partition_method == 'cc':
            messages.find_connected_components(self.conn)

        begin = time.time()

        # Open a cursor to perform database operations
        cur = self.conn.cursor()

        print(len(partition))
        print(partition[0].keys())

        # Define functions that sql needs
        for op in partition[0]["sql_prefix"]:
            cur.execute(op)
        # Make the changes to the database persistent
        self.conn.commit()

        # Select which partitioning scheme to use
        if self.partition_type is not None:
            # Type was prespecified
            for p in partition:
                if p["partition_types"] == self.partition_type:
                    p0 = p
        else:
            # Evaluating costs
            print(80 * "*")
            optimal_cost = None
            for p in partition:
                cur.execute(p["sql_to_cost"])
                cost = cur.fetchone()[0]
                print('Partition scheme "' + p["partition_types"] +
                      '" has cost ' + str(cost))
                if optimal_cost is None or cost < optimal_cost:
                    optimal_cost = cost
                    p0 = p
            print(80 * "*")

        # p0 is partition to use
        for k in p0.keys():
            print(k)
            print(p0[k])
            print()

        # This adds partition information to the database
        print("Running sql_to_apply")
        sql_to_apply_begin = time.time()
        for op in p0["sql_to_apply"]:
            # Currently ignoring the column already exists error generated
            # from ALTER statements
            # TODO: better fix?
            try:
                cur.execute(op)
                # Make the changes to the database persistent
                self.conn.commit()
            except psycopg2.ProgrammingError:
                print("Unexpected error:", sys.exc_info())
                self.conn.rollback()
        sql_to_apply_end = time.time()
        print("Done running sql_to_apply: " +
              str(sql_to_apply_end - sql_to_apply_begin))
        # Grab factor graph data
        self.get_fg(cur)
        # Close communication with the database
        cur.close()
        self.conn.close()

    def load_own_fg(self):
        """TODO."""
        # TODO: this is already in prepare_db
        # Need to refactor this
        # Needs to load the factor graph
        # Track what to sample
        # Track map for variables/factors from each minion
        pass

    def load_minions_fg(self, db_url):
        """TODO."""
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

    def sync_mapping(self):
        """TODO."""
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
                tag, data = evdata['tag'], evdata['data']['data']
                self.map_from_minion[data["pid"]] = \
                    messages.deserialize(data["map"], np.int64)
                resp += 1
        print("DONE WITH SENDING MAPPING")

        for i in range(len(self.map_to_minions)):
            self.map_to_minions[i] = \
                messages.inverse_map(self.vid, self.map_to_minions[i])

        for i in range(len(self.map_from_minion)):
            for j in range(len(self.map_from_minion[i])):
                self.map_from_minion[i][j] = \
                    messages.inverse_map(self.vid, self.map_from_minion[i][j])

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
        """TODO."""
        minion_status = self.runner.cmd('manage.status')
        print("***** MINION STATUS REPORT *****")
        print(minion_status)
        print("UP: ", len(minion_status['up']))
        print("DOWN: ", len(minion_status['down']))
        print()
        return minion_status

    def prep_local_numbskull(self):
        """TODO."""
        self.ns = numbskull.NumbSkull(**vars(self.args))

    def prep_minions_numbskull(self):
        """TODO."""
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
                if data['data']['status'] == 'FAIL':
                    print('ERROR: Minion %s failed to load numbskull.'
                          % data['id'])
                    SUCCESS = False
                resp.append((data['id'], SUCCESS))
        if SUCCESS:
            print('SUCCESS: All minions loaded numbskull.')
        return SUCCESS


def main(application_dir, machines, threads_per_machine,
         learning_epochs, inference_epochs, 
         partition_method, partition_type=None):
    """TODO."""
    # Inputs for experiments:
    #   - dataset
    #   - number of machines
    #   - number of threads per machine
    #   - learning/inference epochs
    #   - sweeps per epoch
    # Return values:
    #   - Time for database
    #   - Time for loading
    #   - Time for learning
    #   - Time for inference
    #   - Memory usage (master, all minions)
    # TODO: how to automate partition selection
    args = ['-l', '1',
            '-i', '1',
            '-t', str(threads_per_machine),
            '-s', '0.01',
            '--regularization', '2',
            '-r', '0.1',
            '--quiet']

    ns_master = NumbskullMaster(application_dir, 
                                machines, 
                                partition_method, 
                                partition_type,
                                args)
    ns_master.initialize()
    learn_time = ns_master.learning(learning_epochs)
    infer_time = ns_master.inference(inference_epochs)

    return ns_master, {"learning_time": learn_time,
                       "inference_time": infer_time}

if __name__ == "__main__":
    if len(sys.argv) == 7 or len(sys.argv) == 8:
        application_dir = sys.argv[1]
        machines = int(sys.argv[2])
        threads_per_machine = int(sys.argv[3])
        learning_epochs = int(sys.argv[4])
        inference_epochs = int(sys.argv[5])
        partition_method = sys.argv[6]
        partition_type = None
        if len(sys.argv) == 8:
            partition_type = sys.argv[7]

        main(application_dir, machines, threads_per_machine,
             learning_epochs, inference_epochs,
             partition_method, partition_type)
    else:
        print("Usage: " + sys.argv[0] +
              " application_dir" +
              " machines" +
              " threads_per_machine" +
              " learning_epochs" +
              " inference_epochs" +
              " partition_method (cc, metis, sp)" +
              " partition_type (type for sp)")
