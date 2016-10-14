#!/bin/bash

for i in `seq 2 10`
do
    ssh raiders${i} "salt-minion -l debug &"
done

