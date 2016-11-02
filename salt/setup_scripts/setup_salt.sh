#!/bin/bash

# Usage: ./setup_salt.sh <index> <master> <root_dir> <interface> <username>

index=''
if [ -n "${1+x}" ]
then
    index=$1
fi

master=raiders6
if [ -n "${2+x}" ]
then
    root_dir=$2
fi

#interface=`curl ifconfig.me`
#interface=`ip -o addr  | grep '\(eth\|p[0-9]p[0-9]\).*inet ' | tr " " "\n" | grep -Eo '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' | head -n 1`
interface=0.0.0.0
if [ -n "${4+x}" ]
then
    interface=$4
fi

user=`whoami`
if [ -n "${5+x}" ]
then
    user=$5
fi

root_dir=/tmp/salt_${user}${index}/
if [ -n "${3+x}" ]
then
    root_dir=$3
fi

id=`hostname`:${user}${index}

cat <<EOL
Configuration
=============
master: ${master}
root_dir: ${root_dir}
interface: ${interface}
user: ${user}
id: ${id}

Add to .bashrc
==============
>> export SALT_ROOT=${root_dir}
>> export SALT_CONFIG_DIR=${root_dir}/etc/salt/
EOL

for i in etc/salt \
         var/cache/run \
         run/salt \
         srv \
         srv/salt \
         srv/salt/_engines \
         srv/salt/_modules \
         srv/pillar \
         srv/salt-master \
         var/log/salt \
         var
do
    mkdir -p ${root_dir}/${i};
done

cp -r salt/conf/* ${root_dir}/etc/salt/

interface=$interface \
user=$user \
root_dir=$root_dir \
envsubst < master > ${root_dir}/etc/salt/master

master=$master \
user=$user \
root_dir=$root_dir \
id=$id \
envsubst < minion > ${root_dir}/etc/salt/minion
