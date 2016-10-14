#!/bin/bash

for i in `seq 1 10`
do
    ssh raiders${i} "rm -rf ${SALT_ROOT}"
done

