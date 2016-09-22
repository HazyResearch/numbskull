#!/bin/bash

cp numbskull_minion.py /tmp/salt/srv/salt/_engines/
salt "*" saltutil.sync_engines
salt "*" service.restart salt-minion
