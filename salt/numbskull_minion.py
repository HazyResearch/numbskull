# -*- coding: utf-8 -*-
'''
A simple test engine, not intended for real use but as an example
'''

# Import python libs
from __future__ import absolute_import
import json


import logging

# Import salt libs
import salt.utils.event

#import numbskull
import sys
import salt.config
__opts__ = salt.config.minion_config('/tmp/salt/etc/salt/minion')
sys.path.append(__opts__['extension_modules']+'/modules')
try:
    pass
    import numbskull
except:
    import time
    for i in range(1000):
          log.debug("FAIL")
          time.sleep(1)


log = logging.getLogger(__name__)

class Counter():
    def __init__(self):
        self.count = 0.0

    def increase(self,x):
        self.count += x

def start():
    '''
    Listen to events and write them to a log file
    '''

    counter = Counter()
    minion = False

    if __opts__['__role'] == 'master':
        event_bus = salt.utils.event.get_master_event(
                __opts__,
                __opts__['sock_dir'],
                listen=True)
        log.debug('test master engine started')
    else:
        event_bus = salt.utils.event.get_event(
            'minion',
            transport=__opts__['transport'],
            opts=__opts__,
            sock_dir=__opts__['sock_dir'],
            listen=True)
        minion = True
        log.debug('test engine started')
    log.debug(minion)
    while True:
        log.debug("NUMBSKULL 123")
        log.debug(numbskull.__file__)
        evdata = event_bus.get_event(full=True)
        if evdata:
           tag, data = evdata['tag'], evdata['data']
           jevent = json.dumps(data)
           log.debug(jevent)
           if data:
              if minion and tag=='incr':
                  counter.increase(data["incr"])
                  tag = 'minion_response'
                  data = {'count':counter.count}
                  log.debug('in if')
                  log.debug(str(counter.count))
                  __salt__['event.send'](tag,data)
