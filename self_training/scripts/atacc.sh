#!/bin/bash

cd $(dirname $0)/../..

export CUDA_VISIBLE_DEVICES=$1
TMPDIR=/tmp
data=weather
model=bart_large
testpfx=test

hyp=self_training/tmp.atacc/fwd

genpfx=gen.$testpfx

tmp=$(mktemp -d autotreeacc.XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX)
prep=self_training/data-prep/$data
sed 's/\[\S\+//g;s/\]//g' $hyp | awk '{$1=$1;print}' > $tmp/test.ar-mr.ar
python bpe_encoder/multiprocessing_bpe_encoder.py \
  --encoder-json "$prep/common/encoder.weather.json" \
  --vocab-bpe    "$prep/common/vocab.gpt2.bpe"       \
  --inputs       "$tmp/test.ar-mr.ar"                \
  --outputs      "$tmp/test.bpe.ar-mr.ar"            \
  --workers      60                                  \
  --keep-empty
ln -s "../self_training/data-prep/$data/common/dict.gpt2.txt" "$tmp/dict.mr.txt"
ln -s "../self_training/data-prep/$data/common/dict.gpt2.txt" "$tmp/dict.ar.txt"
REVMDLDIR=self_training/checkpoints/$data/pct-1c/shuf.lbl.rrk-rev.rev.$model
fairseq-generate $tmp \
  --user-dir . \
  --encoder-json data-prep/$data/encoder.weather.json \
  --vocab-bpe data-prep/$data/vocab.gpt2.bpe \
  --gen-subset test.bpe \
  --path $REVMDLDIR/checkpoint_best.pt \
  --dataset-impl raw \
  --max-tokens 2048 \
  --beam 5 \
  --max-len-a 2 --max-len-b 50 \
  > $tmp/gen.rev.bpe.txt
grep ^H- $tmp/gen.rev.bpe.txt | sort -n -k 2 -t - | awk -F '\t' '{print $3}' > $tmp/hyp.bpe.txt
python scripts/gpt2_bpe_decoder.py \
  data-prep/$data/encoder.$data.json \
  data-prep/$data/vocab.gpt2.bpe \
  $tmp/hyp.bpe.txt \
  $tmp/hyp.norm.txt
python scripts/compute_tree_acc.py $tmp/hyp.norm.txt data-prep/$data/$testpfx.mr-ar.mr
rm -rf $tmp
