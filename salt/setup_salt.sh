#!/bin/bash

# Usage: ./setup_salt.sh <master> <root_dir> <interface> <username>

# Default master to raiders1
master=172.24.75.13
if [ -n "${1+x}" ]
then root_dir=$1
fi

root_dir=/tmp/salt/
if [ -n "${2+x}" ]
then
    root_dir=$2
fi

#interface=`curl ifconfig.me`
interface=`ip -o addr  | grep '\(eth\|p[0-9]p[0-9]\).*inet ' | tr " " "\n" | grep -Eo '[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}' | head -n 1`
if [ -n "${3+x}" ]
then
    interface=$3
fi

user=`whoami`
if [ -n "${4+x}" ]
then
    user=$4
fi

id=`hostname`:${user}

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
