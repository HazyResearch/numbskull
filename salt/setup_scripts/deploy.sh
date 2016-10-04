#!/bin/bash

cp ../src/numbskull_minion.py ${SALT_ROOT}/srv/salt/_engines/
cp -r ../../numbskull ${SALT_ROOT}/srv/salt/_modules/
cp ../src/messages.py ${SALT_ROOT}/srv/salt/_modules/
salt "*" saltutil.sync_engines
salt "*" saltutil.sync_modules
#salt "*" service.restart salt-minion
