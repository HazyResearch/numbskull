# Import python libs
from __future__ import absolute_import
import json
import logging
import os.path
import numbskull
from numbskull import numbskull

# Import salt libs
import salt.utils.event
import salt.client
import salt.config
import nb_syspaths as syspaths



class NumbskullServer:
    def __init__(self, args):
        self.salt_master_conf_dir = os.path.join(syspaths.CONFIG_DIR, 'master')
        self.salt_master = salt.client.LocalClient(self.salt_master_conf_dir)
        self.salt_opts = salt.config.client_config(self.salt_master_conf_dir)
        self.event_bus = salt.utils.event.get_event('master',
                         sock_dir=self.salt_opts['sock_dir'],
                         transport=self.salt_opts['transport'],
                         opts=self.salt_opts)
        self.sampler = numbskull.load(args)

    def learning(self):
        self.sampler.learning()

    def inference(self):
        self.sampler.inference()

    def start(self):
        for i in range(10):
            # send message to minion TODO: Change client
            data = {'incr':10}
            tag = 'incr'
            newEvent = self.salt_master.cmd('raiders2_thodrek','event.fire',[data, tag])
            # receive message from minion
            evdata = self.event_bus.get_event(wait=10,tag='minion_response',full=True)
            if not evdata:
               print 'Nothing back'
            # print message
            if evdata:
                tag, data = evdata['tag'], evdata['data']
                jevent = json.dumps(data)
                print 'Received new message:' + jevent
                print 

def main(argv=None):
    args = ['test',
        '-l', '100',
        '-i', '100',
        '-t', '10',
        '-s', '0.01',
        '--regularization', '2',
        '-r', '0.1',
        '--quiet']

    nsServer = NumbskullServer(args)
