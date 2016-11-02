#!/bin/bash

for i in `seq 1 10`
do
    ssh raiders${i} "cd `pwd`; ./setup_salt.sh; for index in \`seq 1 4\`; do ./setup_salt.sh \${index}; done"
done

