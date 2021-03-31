#!/bin/bash

cd $(dirname $0)/../..

pct=$1
maxitr=$2
gpu=$3

srp=self_training/scripts

#$srp/train.weather.bart.sh $pct 0 lbl $gpu

for itr in $(seq 1 $maxitr); do
  $srp/prepare.weather.plbl.constr.sh $pct $itr $gpu
  $srp/train.weather.bart.constr.sh   $pct $itr plbl $gpu
  $srp/train.weather.bart.constr.sh   $pct $itr lbl  $gpu
done
