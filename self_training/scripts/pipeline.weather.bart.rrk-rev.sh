#!/bin/bash

cd $(dirname $0)/../..

pct=$1
maxitr=$2
gpu=$3

srp=self_training/scripts

#$srp/train.weather.bart.sh $pct 0 lbl $gpu

$srp/prepare.weather.rev.sh $pct
for itr in $(seq 1 $maxitr); do

#itr=3

  itrprev=$(bc <<< "$itr-1")
  $srp/prepare.weather.plbl.rrk-rev.rev.sh $pct $itrprev
  $srp/train.weather.bart.rrk-rev.rev.sh   $pct $itrprev plbl $gpu
  $srp/train.weather.bart.rrk-rev.rev.sh   $pct $itrprev lbl  $gpu
  $srp/prepare.weather.plbl.rrk-rev.sh     $pct $itr     $gpu
  $srp/train.weather.bart.rrk-rev.sh       $pct $itr     plbl $gpu
  $srp/train.weather.bart.rrk-rev.sh       $pct $itr     lbl  $gpu

done
