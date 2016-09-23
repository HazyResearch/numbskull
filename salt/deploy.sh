#!/bin/bash

cp numbskull_minion.py /tmp/salt/srv/salt/_engines/
cp -r ../numbskull /tmp/salt/srv/salt/_modules/
salt "*" saltutil.sync_engines
salt "*" saltutil.sync_modules
#salt "*" service.restart salt-minion
