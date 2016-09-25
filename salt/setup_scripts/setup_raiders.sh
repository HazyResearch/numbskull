#!/bin/bash

for i in `seq 1 10`
do
    ssh raiders${i} "cd `pwd`; ./setup_salt.sh"
done

