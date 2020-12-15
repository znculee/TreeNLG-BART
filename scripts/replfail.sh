#!/bin/bash

if [ $# -eq 3 ]; then
  gen=$(readlink -f $1)
  hyp=$(readlink -f $2)
  base=$(readlink -f $3)
else
  echo "Usage: replfail.sh generations hypothesis base"
  exit 0
fi

cd $(dirname $0)/..

repl=$(grep ^H- $gen | awk -F '\t' '$2=="-inf" {print $1}' | cut -d '-' -f 2 | awk '{print $1+1}')
awk -F '\t' 'NR==FNR {l[$0];next} !(FNR in l) {print $1} (FNR in l) {print $2}' \
  <(echo "$repl") \
  <(paste $hyp $base)
