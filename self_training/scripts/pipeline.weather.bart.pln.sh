#!/bin/bash

cd $(dirname $0)/../..

pct=$1
maxitr=$2
gpu=$3

srp=self_training/scripts

#$srp/prepare.weather.pln.sh $pct
$srp/train.weather.bart.pln.sh $pct 0 lbl $gpu

for itr in $(seq 1 $maxitr); do
  $srp/prepare.weather.plbl.pln.sh $pct $itr $gpu
  $srp/train.weather.bart.pln.sh   $pct $itr plbl $gpu
  $srp/train.weather.bart.pln.sh   $pct $itr lbl  $gpu
done
