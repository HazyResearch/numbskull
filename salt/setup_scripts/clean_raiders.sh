#!/bin/bash -x

# Example usage:
# >> ./clean_raiders.sh '/tmp/salt_bryanhe*'

dir=${SALT_ROOT}
if [ -n "${1+x}" ]
then
    dir=$1
fi

for i in `seq 1 10`
do
    ssh raiders${i} "rm -rf ${dir}"
done

