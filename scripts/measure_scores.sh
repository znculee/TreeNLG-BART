#!/bin/bash

if [ $# -eq 2 ]; then
  hyp=$(readlink -f $1)
  ref=$(readlink -f $2)
else
  echo "Usage: measure_scores.sh hypothesis reference"
  exit 0
fi

cd $(dirname $0)/..

scorer=e2e-metrics/measure_scores.py
tmp=$(mktemp -d measure_scores.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX)
hyprmt=$tmp/hyprmt
refrmt=$tmp/refrmt

mkdir -p $tmp

rmtreeinfo () {
  sed 's/\[\S\+//g;s/\]//g' | awk '{$1=$1;print}'
}

cat $ref | rmtreeinfo > $refrmt
cat $hyp | rmtreeinfo > $hyprmt

python $scorer -p $refrmt $hyprmt 2> /dev/null

rm -rf $tmp
