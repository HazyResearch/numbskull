#!/bin/bash

git clone git@github.com:saltstack/salt.git
cd salt/
pip install -r requirements/base.txt --user
python setup.py install --user

